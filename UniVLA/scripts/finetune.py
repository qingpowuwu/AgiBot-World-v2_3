import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import argparse
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.utils.rnn import pad_sequence
import tqdm
from accelerate import PartialState, Accelerator
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from accelerate import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForVision2Seq, 
    AutoProcessor, 
    BitsAndBytesConfig, 
    AutoConfig, 
    AutoImageProcessor,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.policy.transformer_utils import MAPBlock
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_Geniesim
import prismatic.vla.datasets.pretrainAe_a2d_pretrain_v6 as a2d_cfg

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ActionDecoder(torch.nn.Module):
    def __init__(
        self,
        n_layers=1,
        vis_dim=4096,
        n_joints=16,
        window_size=30,
        hidden_dim=512,
        with_proprio=False,
        wogripper=False,
        ):
        super().__init__()
        
        self.with_proprio = with_proprio
        self.wogripper = wogripper
        
        if with_proprio:
            if wogripper:
                self.proprio_proj = nn.Linear(n_joints-2, hidden_dim)  # remove gripper
            else:
                self.proprio_proj = nn.Linear(n_joints, hidden_dim)
            
        self.proj_l = nn.Linear(2176, vis_dim)
        self.proj_r = nn.Linear(2176, vis_dim)
        self.proj_h = nn.Linear(2176, vis_dim)
        
        self.latent_action_pool = MAPBlock(
            n_layers=n_layers,
            vis_dim=vis_dim,
            embed_dim=hidden_dim,
            n_heads=hidden_dim//64,
            )
        
        self.visual_pool = MAPBlock(
            vis_dim=vis_dim,
            embed_dim=hidden_dim,
            n_heads=hidden_dim//64,
            )
        
        if with_proprio:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim * 8), 
                nn.GELU(),
                nn.Linear(hidden_dim * 8, n_joints * window_size),
            )
        else:
            self.proj = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 8), 
                nn.GELU(),
                nn.Linear(hidden_dim * 8, n_joints * window_size),
            )
        
    def forward(self, latent_action_tokens, visual_embed, raw_visual, proprio=None):
        
        visual_embed = torch.cat((visual_embed, self.proj_h(raw_visual[:,:256,:]), self.proj_l(raw_visual[:,256:512,:]), self.proj_r(raw_visual[:,512:768,:])),dim=1)
        visual_embed = self.visual_pool(visual_embed)
        
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed=visual_embed)
        
        if proprio is not None:
            proprio = proprio.squeeze(1)
            if self.wogripper:
                proprio_l_arm = proprio[:,:7]
                proprio_r_arm = proprio[:,8:-1]
                proprio = torch.concat((proprio_l_arm, proprio_r_arm), dim=-1)
            proprio = self.proprio_proj(proprio)
            action = self.proj(torch.cat((action_token, proprio), dim=1))
        else:
            action = self.proj(action_token)

        return action
    
    
class Wrapped_Model(torch.nn.Module):
    def __init__(
        self,
        vla,
        freeze_vla=True,
        window_size=30,
        decoder_n_layers=1,
        decoder_hidden_dim=512,
        with_proprio=False,
        wogripper=False,
        action_decoder_path="",
        decoupled_loss=False,
        ):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        self.action_decoder = ActionDecoder(
            n_layers=decoder_n_layers,
            hidden_dim=decoder_hidden_dim,
            with_proprio=with_proprio,
            wogripper=wogripper,
            window_size=window_size,
            )
        try:
            self.action_decoder.load_state_dict(torch.load(action_decoder_path))
            print("success loading decoder checkpoint")
        except:
            pass
        
        self.decoupled_loss = decoupled_loss
        self.with_proprio = with_proprio
        if freeze_vla:
            self.vla.requires_grad_(False)
            

    def forward(self, batch):
        slow_output = self.slow_forward(batch)
        loss, loss_one_step, latent_action_tokens = self.fast_forward(batch, slow_output)

        return slow_output, loss, loss_one_step, latent_action_tokens


    def slow_forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output: CausalLMOutputWithPast = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states = True,        # Return intermediate tokens of all layers
            )
        return output
    

    def fast_forward(self, batch, slow_output):
        # Task and action latents
        visual_embed = slow_output.hidden_states[-1][:, :self.vla.vision_backbone.featurizer.patch_embed.num_patches].to(torch.float)

        latent_tokens = slow_output.hidden_states[-1][:, self.vla.vision_backbone.featurizer.patch_embed.num_patches: ]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt > 32000

        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)

        # Run specialist policy
        if self.with_proprio:
            proprio = batch['proprio']
        else:
            proprio = None
        

        raw_visual = slow_output.hidden_states[0]
        
        pred_action = self.action_decoder(latent_action_tokens, visual_embed, raw_visual, proprio).reshape(-1, self.window_size, 16)
        
        if self.decoupled_loss:
            loss_l_arm = torch.nn.functional.smooth_l1_loss(torch.nn.functional.tanh(pred_action[..., :7]), batch['actions'][..., :7], reduction='none', beta=0.1)
            loss_r_arm = torch.nn.functional.smooth_l1_loss(torch.nn.functional.tanh(pred_action[..., 8:-1]), batch['actions'][..., 8:-1], reduction='none', beta=0.1)
            loss_arm = loss_l_arm + loss_r_arm
            gripper_l_label = (batch['actions'][..., 7] > 0.5).float()
            gripper_r_label = (batch['actions'][..., -1] > 0.5).float()
            loss_l_gripper = torch.nn.functional.binary_cross_entropy_with_logits(pred_action[..., 7], gripper_l_label)
            loss_r_gripper = torch.nn.functional.binary_cross_entropy_with_logits(pred_action[..., -1], gripper_r_label)
            loss_gripper = loss_l_gripper + loss_r_gripper
            loss_one_step = loss_arm[:,0].mean()
            loss = loss_arm.mean() + loss_gripper
        else:
            loss = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
            loss_one_step = loss[:,0].mean()
            loss = loss.mean()

        return loss, loss_one_step, latent_action_tokens


def finetune(cfg):
    
    # [Validate] Ensure GPU Available & Set Device / Distributed Context
    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(mixed_precision="bf16", kwargs_handlers=[ddp_kwargs])

    # Start =>> Build Directories
    run_dir, adapter_dir = cfg.run_root_dir, cfg.adapter_tmp_dir
    os.makedirs(run_dir, exist_ok=True)

    # Quantization Config =>> only if LoRA fine-tuning
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Load OpenVLA Processor and Model using HF AutoClasses
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    model = Wrapped_Model(
        vla = vla,
        freeze_vla=cfg.freeze_vla,
        window_size=cfg.window_size,
        decoder_n_layers=cfg.decoder_n_layers,
        decoder_hidden_dim=cfg.decoder_hidden_dim,
        with_proprio=cfg.with_proprio,
        wogripper=cfg.wogripper,
        action_decoder_path=cfg.adr_path,
        decoupled_loss=cfg.decouple,
        ).to(device_id)

    trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)
    
    # Create Optimizer =>> note that we default to a simple constant learning rate!
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(cfg.max_steps * 8 * 0.8), gamma=0.1)

    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel
    
    latent_action_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.,
    )

    cpu_state = torch.load(cfg.lam_path, map_location="cpu")["state_dict"]
    for name, param in latent_action_model.named_parameters():
        param.data = cpu_state["lam." + name].to(device_id, non_blocking=True)
        del cpu_state["lam." + name]  # 及时释放 CPU 副本
    torch.cuda.empty_cache()
    latent_action_model.eval()
    
    # Load gensim dataset
    from prismatic.vla.datasets import A2dDataset
    train_set = {}
    val_set = {}
    for num in cfg.task_ids:
        train_set[str(num)] = {
            "use_cam_list": ["head", "hand_right", "hand_left"],
            "label_file_name": f"task_{num}_train.json",
        }
        val_set[str(num)] = {
            "use_cam_list": ["head", "hand_right", "hand_left"],
            "label_file_name": f"task_{num}_val.json",
        }
    dataset_args = a2d_cfg.DatasetArguments(
        meta_json_dir=cfg.meta_json_dir,
        data_root_dir=cfg.data_root_dir,
        dataset_task_cfg=train_set,
        eval_dataset_task_cfg=val_set,
    )
    data_training_args = a2d_cfg.DataTrainingArguments(force_image_size=224)
    ActionSpacePadder = a2d_cfg.ActionSpacePadderArguments()

    vla_dataset = A2dDataset(
        # base parmas
        label_file_dir=dataset_args.meta_json_dir, 
        data_root_dir=dataset_args.data_root_dir, 
        valid_episode_txt=dataset_args.valid_episode_txt, 
        world_size=dist.get_world_size(), 
        rank_id=dist.get_rank(), 
        sample_rate=dataset_args.train_sample_rate, 
        online_process_mp_cnt=dataset_args.online_process_mp_cnt, 
        # a2d params
        num_image_token=int((dataset_args.force_image_size // 14) ** 2 * (0.5**2)), 
        is_train=True, 
        image_size=data_training_args.force_image_size, 
        pad2square=data_training_args.pad2square, 
        dynamic_image_size=data_training_args.dynamic_image_size, 
        use_thumbnail=data_training_args.use_thumbnail, 
        min_dynamic_patch=data_training_args.min_dynamic_patch, 
        max_dynamic_patch=data_training_args.max_dynamic_patch, 
        normalize_type=data_training_args.normalize_type, 
        action_chunk_size=cfg.window_size,
        use_real_state=True, 
        conversation_type=data_training_args.conversation_type, 
        vis_frame=False, 
        vis_dir="", 
        ActionSpacePadder=ActionSpacePadder, 
        min_window_size=cfg.window_size, 
        max_window_size=cfg.window_size + 1, 
        image_transform=processor.image_processor.apply_transform,
    )

    vla_dataset.generate_task_infos(
        dataset_args.dataset_task_cfg,
        task_episode_processors_cfg=dataset_args.episode_processors,
        task_dataset_processors_cfg=dataset_args.dataset_processors,
        task_runtime_processors_cfg=dataset_args.runtime_processors,
        shuffle=True,
        statistic=True,
        debug_one_episode=cfg.debug,
        # debug_one_episode=False,
    )

    collator = PaddedCollatorForActionPrediction_Geniesim()
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        shuffle=True,
        collate_fn=collator,
        pin_memory=False,
        num_workers=64,
    )
    
    model, latent_action_model, optimizer, scheduler, dataloader = accelerator.prepare(
        model, latent_action_model, optimizer, scheduler, dataloader
    )

    # Initialize Logging =>> W&B
    if distributed_state.is_main_process and not cfg.debug:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project)

    # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        model.train()
        optimizer.zero_grad()
        current_step = 0
        if distributed_state.is_main_process:
            # summary writter
            writer = SummaryWriter(log_dir=cfg.run_root_dir)
                
        for e in range(10000):
            progress.set_description("Epoch " + str(e+1))
                
            for batch_idx, batch in enumerate(dataloader):
                batch["init_pixel_values"] = batch["init_pixel_values"].to(device_id) # [8, 3, 224, 224]
                batch["goal_pixel_values"] = batch["goal_pixel_values"].to(device_id) # [8, 3, 224, 224]
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id) # [8, 6, 224, 224]
                batch['actions'] = batch['actions'].to(device_id) # [8, 12, 7]
                batch['proprio'] = batch['proprio'].to(device_id) # [8, 7]
           
                if len(batch["hist_init_pixel_values"]) > 1:
                    batch["hist_init_pixel_values"] = batch["hist_init_pixel_values"].to(device_id) # [2, 3, 224, 224]
                    batch["hist_goal_pixel_values"] = batch["hist_goal_pixel_values"].to(device_id) # [2, 3, 224, 224]

                    with torch.no_grad():
                        video = torch.stack([batch["init_pixel_values"], batch["goal_pixel_values"]], dim=1) # [8, 2, 3, 224, 224]
                        latent_action_idx_batch = latent_action_model.module.vq_encode(video)['indices'].squeeze() # [8, 4]
                        video = torch.stack([batch["hist_init_pixel_values"], batch["hist_goal_pixel_values"]], dim=1) # [2, 2, 3, 224, 224]
                        latent_action_idx_history = latent_action_model.module.vq_encode(video)['indices'].squeeze() # [2, 4]

                    input_ids_list = []
                    labels_list = []
                    hist_idx = 0
                    
                    if batch['actions'].shape[0] == 1:
                        latent_action_idx_batch = latent_action_idx_batch.unsqueeze(0)
                    
                    for idx, latent_action_idx in enumerate(latent_action_idx_batch):
                        action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]   # [ACT_1, ACT_2, ... ACT_K]
                        action_tokens = ''
                        for i, action in enumerate(action_vocab):
                            action_tokens += action
                        
                        if batch['with_hist'][idx]:
                            action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx_history[hist_idx]]

                            hist_action_tokens = ''
                            for i, action in enumerate(action_vocab):
                                hist_action_tokens += action

                            input_prompt = f"What action should the robot take to {batch['instructions'][idx]}? History action " + hist_action_tokens
                            hist_idx += 1
                        else:
                            input_prompt = f"What action should the robot take to {batch['instructions'][idx]}?"

                        # print(input_prompt)
                        # Add instruction to VLA prompt
                        prompt_builder = PurePromptBuilder("openvla")
                        conversation = [
                            {"from": "human", "value": input_prompt},
                            {"from": "gpt", "value": action_tokens},
                        ]
                        for turn in conversation:
                            prompt_builder.add_turn(turn["from"], turn["value"])

                        # Tokenize (w/ `base_tokenizer`)
                        input_ids = processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
                        labels = list(input_ids)

                        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
                        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
                        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

                        labels[: -(len(action_vocab) + 1)] = -100

                        input_ids_list.append(input_ids)
                        labels_list.append(labels)
                
                else:
                    with torch.no_grad():
                        video = torch.stack([batch["init_pixel_values"], batch["goal_pixel_values"]], dim=1)
                        latent_action_idx_batch = latent_action_model.module.vq_encode(video)['indices'].squeeze()

                    input_ids_list = []
                    labels_list = []
                    
                    if batch['actions'].shape[0] == 1:
                        latent_action_idx_batch = latent_action_idx_batch.unsqueeze(0)
                        
                    for idx, latent_action_idx in enumerate(latent_action_idx_batch):
                        action_vocab = [f'<ACT_{i.item()}>' for i in latent_action_idx]   # [ACT_1, ACT_2, ... ACT_K]

                        action_tokens = ''
                        for i, action in enumerate(action_vocab):
                            action_tokens += action

                        # Add instruction to VLA prompt
                        prompt_builder = PurePromptBuilder("openvla")
                        conversation = [
                            {"from": "human", "value": f"What action should the robot take to {batch['instructions'][idx]}?"},
                            {"from": "gpt", "value": action_tokens},
                        ]
                        for turn in conversation:
                            prompt_builder.add_turn(turn["from"], turn["value"])

                        # Tokenize (w/ `base_tokenizer`)
                        input_ids = processor.tokenizer(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
                        labels = list(input_ids)

                        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
                        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
                        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)

                        labels[: -(len(action_vocab) + 1)] = -100

                        input_ids_list.append(input_ids)
                        labels_list.append(labels)
            
                input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
                labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

                # Truncate (if necessary)
                input_ids, labels = input_ids[:, : processor.tokenizer.model_max_length], labels[:, : processor.tokenizer.model_max_length]

                # Get `attention_mask` by checking for `pad_token_id`
                attention_mask = input_ids.ne(processor.tokenizer.pad_token_id)

                batch["input_ids"] = input_ids
                batch["attention_mask"] = attention_mask
                batch["labels"] = labels

                output, act_loss, loss_one_step, latent_action_proj = model(batch)
                loss = act_loss if cfg.freeze_vla else act_loss + (output.loss) * cfg.lam_loss_weight
                normalized_loss = loss / cfg.grad_accumulation_steps

                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.3)
                # Backward pass
                normalized_loss.backward()

                # Compute Accuracy and L1 Loss for Logging
                action_logits = output.logits[:, model.module.vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
                action_preds = action_logits.argmax(dim=2)
                action_gt = batch["labels"][:, 1:].to(action_preds.device)
                mask = action_gt > 32000

                # Compute Accuracy
                correct_preds = (action_preds == action_gt) & mask
                action_accuracy = correct_preds.sum().float() / mask.sum().float()


                # Store recent train metrics
                recent_losses.append(loss.item())
                recent_action_accuracies.append(action_accuracy.item())

                # Compute gradient step index
                gradient_step_idx = batch_idx // cfg.grad_accumulation_steps

                # Compute smoothened train metrics
                #   =>> Equal to current step metrics when not using gradient accumulation
                #   =>> Otherwise, equal to the average of metrics observed over micro-batches used for gradient accumulation
                smoothened_loss = sum(recent_losses) / len(recent_losses)
                smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)

                # Push Metrics to W&B (every 10 gradient steps)
                # if distributed_state.is_main_process and gradient_step_idx % 5 == 0 and not cfg.debug:
                    
                #     wandb.log(
                #         {
                #             "train_loss": smoothened_loss,
                #             "latent_action_accuracy": smoothened_action_accuracy,
                #             "action_loss": act_loss.item(),
                #             "action_loss_1step": loss_one_step.item(),
                #             "lr": optimizer.state_dict()['param_groups'][0]['lr']
                #             # "latent_align_loss": latent_align_loss.item(),
                #         },
                #         step=gradient_step_idx + current_step,
                #     )

                # Initialize Logging =>> TensorBoard
                if distributed_state.is_main_process:
                    # add_scalar
                    writer.add_scalar('train_loss', smoothened_loss, gradient_step_idx + current_step)
                    writer.add_scalar('latent_action_accuracy', smoothened_action_accuracy, gradient_step_idx + current_step)
                    writer.add_scalar('action_loss', act_loss.item(), gradient_step_idx + current_step)
                    writer.add_scalar('action_loss_1step', loss_one_step.item(), gradient_step_idx + current_step)
                    writer.add_scalar('lr', optimizer.state_dict()['param_groups'][0]['lr'], gradient_step_idx + current_step)
                    # writer.add_scalar('latent_align_loss', latent_align_loss.item(), gradient_step_idx + current_step)
                    
                # Optimizer Step
                if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    progress.update()

                # Save Model Checkpoint =>> by default, only keeps the latest checkpoint, continually overwriting it!
                if (gradient_step_idx + current_step) > 0 and (gradient_step_idx + current_step) % cfg.save_steps == 0:
                    if distributed_state.is_main_process:
                        print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                        # If LoRA, we first save adapter weights, then merge into full model; otherwise, default save!
                        save_dir = adapter_dir if cfg.use_lora else run_dir

                        # Save Processor & Weights
                        # if not cfg.freeze_vla:
                        processor.save_pretrained(run_dir)
                        model.module.vla.save_pretrained(save_dir)

                        # Save low-level policy
                        torch.save(model.module.action_decoder.state_dict(), str(run_dir) + f'/action_decoder.pt')

                    # Wait for processor and adapter weights to be saved by main process
                    dist.barrier()

                    # Merge LoRA weights into model backbone for faster inference
                    #   =>> Note that merging is slow and can be done post-hoc to speed up training
                    if cfg.use_lora:
                        base_vla = AutoModelForVision2Seq.from_pretrained(
                            cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                        )
                        merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                        merged_vla = merged_vla.merge_and_unload()
                        if distributed_state.is_main_process:
                            if cfg.save_latest_checkpoint_only:
                                # Overwrite latest checkpoint
                                merged_vla.save_pretrained(run_dir)

                                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                            else:
                                # Prepare to save checkpoint in new directory
                                checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_ckpt")
                                os.makedirs(checkpoint_dir, exist_ok=True)

                                # Save processor and model weights to new directory
                                processor.save_pretrained(checkpoint_dir)
                                merged_vla.save_pretrained(checkpoint_dir)

                                print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

                    # Block on Main Process Checkpointing
                    dist.barrier()
                
            current_step += gradient_step_idx
            # Stop training when max_steps is reached
            if current_step >= cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Config")
    
    # add args
    parser.add_argument("--vla_path", type=str, default="checkpoints/finetuned", help="Path to univla model ckpt")
    parser.add_argument("--lam_path", type=str, default="checkpoints/lam-stage-2.ckpt", help="Path to LAM model ckpt")
    parser.add_argument("--data_root_dir", type=str, default="", help="Path to dataset")
    parser.add_argument("--meta_json_dir", type=str, default="", help="Path to dataset meta json")
    parser.add_argument("--run_root_dir", type=str, default="runs", help="Path to directory to store logs & checkpoints")
    parser.add_argument("--adapter_tmp_dir", type=str, default="adapter-tmp", help="Temporary directory for LoRA weights before fusing")
    parser.add_argument("--batch_size", type=int, default=8, help="Fine-tuning batch size")
    parser.add_argument("--max_steps", type=int, default=5000, help="Max number of fine-tuning steps")
    parser.add_argument("--save_steps", type=int, default=1000, help="Interval for checkpoint saving")
    parser.add_argument("--learning_rate", type=float, default=1.5e-4, help="Fine-tuning learning rate")
    parser.add_argument("--grad_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--image_aug", action="store_true", help="Whether to train with image augmentations")
    parser.add_argument("--shuffle_buffer_size", type=int, default=10000, help="Dataloader shuffle buffer size (can reduce if OOM)")
    parser.add_argument("--save_latest_checkpoint_only", action="store_true", help="Whether to save only one checkpoint per run and continually overwrite the latest checkpoint")
    parser.add_argument("--codebook_size", type=int, default=16, help="Codebook size")
    parser.add_argument("--lam_model_dim", type=int, default=768, help="LAM model dimension")
    parser.add_argument("--lam_latent_dim", type=int, default=128, help="LAM latent dimension")
    parser.add_argument("--lam_num_latents", type=int, default=32, help="LAM number of latents")
    parser.add_argument("--lam_patch_size", type=int, default=14, help="LAM patch size")
    parser.add_argument("--lam_enc_blocks", type=int, default=12, help="LAM encoder blocks")
    parser.add_argument("--lam_dec_blocks", type=int, default=12, help="LAM decoder blocks")
    parser.add_argument("--lam_num_heads", type=int, default=12, help="LAM number of heads")
    parser.add_argument("--window_size", type=int, default=30, help="Window size")
    parser.add_argument("--lam_loss_weight", type=float, default=1, help="LAM loss weight")
    parser.add_argument("--freeze_vla", action="store_true", help="Whether to freeze VLA")
    parser.add_argument("--use_lora", action="store_true", help="Whether to use LoRA fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=32, help="Rank of LoRA weight matrix")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="Dropout applied to LoRA weights")
    parser.add_argument("--use_quantization", action="store_true", help="Whether to 4-bit quantize VLA for LoRA fine-tuning")
    parser.add_argument("--wandb_project", type=str, default="univla-geniesim", help="Name of W&B project to log to")
    parser.add_argument("--wandb_entity", type=str, default="", help="Name of entity to log under")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--decoder_n_layers", type=int, default=1, help="Number of decoder layers")
    parser.add_argument("--decoder_hidden_dim", type=int, default=512, help="Decoder hidden dimension")
    parser.add_argument("--with_proprio", action="store_true", help="Whether to use proprioceptive data")
    parser.add_argument("--wogripper", action="store_true", help="Whether to not use proprioceptive gripper data")
    parser.add_argument("--use_lam_cache", action="store_true", help="Whether to use LAM cache")
    parser.add_argument("--decouple", action="store_true", help="Whether to decouple")
    parser.add_argument("--adr_path", type=str, default="", help="Path to ADR")
    parser.add_argument("--task_ids", nargs='+', type=int, default=[1], help="List of task IDs")

    args = parser.parse_args()
    
    finetune(args)
