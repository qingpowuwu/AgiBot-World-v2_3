import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))
from PIL import Image
from experiments.robot.geniesim.genie_model import WrappedGenieEvaluation, WrappedModel
import rclpy
import threading
from cv_bridge import CvBridge
import cv2
from genie_sim_ros import SimROSNode
import numpy as np
from dataclasses import dataclass
from typing import Union
import draccus


def resize_img(img, width, height):
    resized_img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    resized_img = np.array(resized_img)
    return resized_img


def get_instruction(task_name):

    if task_name == "iros_clear_the_countertop_waste":
        lang = "Pick up the yellow functional beverage can on the table with the left arm.;Threw the yellow functional beverage can into the trash can with the left arm.;Pick up the green carbonated beverage can on the table with the right arm.;Threw the green carbonated beverage can into the trash can with the right arm."
    elif task_name == "iros_restock_supermarket_items":
        lang = "Pick up the brown plum juice from the restock box with the right arm.;Place the brown plum juice on the shelf where the brown plum juice is located with the right arm."
    elif task_name == "iros_clear_table_in_the_restaurant":
        lang = "Pick up the bowl on the table near the right arm with the right arm.;Place the bowl on the plate on the table with the right arm."
    elif task_name == "iros_stamp_the_seal":
        lang = "Pick up the stamp from the ink pad on the table with the right arm.;Stamp the document on the table with the stamp in the right arm.;Place the stamp into the ink pad on the table with the right arm."
    elif task_name == "iros_pack_in_the_supermarket":
        lang = "Pick up the grape juice on the table with the right arm.;Put the grape juice into the felt bag on the table with the right arm."
    elif task_name == "iros_heat_the_food_in_the_microwave":
        lang = "Open the door of the microwave oven with the right arm.;Pick up the plate with bread on the table with the right arm.;Put the plate containing bread into the microwave oven with the right arm.;Push the plate that was not placed properly into the microwave oven the right arm.;Close the door of the microwave oven with the left arm.;Press the start button on the right side of the microwave oven with the right arm."
    elif task_name == "iros_open_drawer_and_store_items":
        lang = "Pull the top drawer of the drawer cabinet with the right arm.;Pick up the Rubik's Cube on the drawer cabinet with the right arm.;Place the Rubik's Cube into the drawer with the right arm.;Push the top drawer of the drawer cabinet with the right arm."
    elif task_name == "iros_pack_moving_objects_from_conveyor":
        lang = "Pick up the hand cream from the conveyor belt with the right arm;Place the hand cream held in the right arm into the box on the table"
    elif task_name == "iros_pickup_items_from_the_freezer":
        lang = "Open the freezer door with the right arm;Pick up the caviar from the freezer with the right arm;Place the caviar held in the right arm into the shopping cart;Close the freezer door with both arms"
    elif task_name == "iros_make_a_sandwich":
        lang = "Pick up the bread slice from the toaster on the table with the right arm;Place the picked bread slice into the plate on the table with the right arm;Pick up the ham slice from the box on the table with the left arm;Place the picked ham slice onto the bread slice in the plate on the table with the left arm;Pick up the lettuce slice from the box on the table with the right arm;Place the picked lettuce slice onto the ham slice in the plate on the table with the right arm;Pick up the bread slice from the toaster on the table with the right arm;Place the bread slice onto the lettuce slice in the plate on the table with the right arm"
    else:
        raise ValueError("task does not exist")

    return lang


def get_sim_time(sim_ros_node):
    sim_time = sim_ros_node.get_clock().now().nanoseconds * 1e-9
    return sim_time


def infer(policy, cfg):
    debug = True  # Set to True to save frames for debugging
    rclpy.init()
    sim_ros_node = SimROSNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
    spin_thread.start()
    init_frame = True
    bridge = CvBridge()
    count = 0
    SIM_INIT_TIME = 10
    action_queue = None

    lang = get_instruction(cfg.task_name)

    while rclpy.ok():
        if action_queue:
            is_end = True if len(action_queue) == 1 else False
            action = action_queue.popleft()
            sim_ros_node.publish_joint_command(action, is_end)

        else:
            img_h_raw = sim_ros_node.get_img_head()
            img_l_raw = sim_ros_node.get_img_left_wrist()
            img_r_raw = sim_ros_node.get_img_right_wrist()
            act_raw = sim_ros_node.get_joint_state()
            infer_start = sim_ros_node.is_infer_start()

            if (init_frame or infer_start) and (
                img_h_raw
                and img_l_raw
                and img_r_raw
                and act_raw
                and img_h_raw.header.stamp
                == img_l_raw.header.stamp
                == img_r_raw.header.stamp
            ):
                sim_time = get_sim_time(sim_ros_node)
                if sim_time > SIM_INIT_TIME:
                    init_frame = False
                    print("cur sim time", sim_time, img_h_raw.header.stamp)
                    count = count + 1
                    img_h = bridge.compressed_imgmsg_to_cv2(
                        img_h_raw, desired_encoding="rgb8"
                    )
                    img_l = bridge.compressed_imgmsg_to_cv2(
                        img_l_raw, desired_encoding="rgb8"
                    )
                    img_r = bridge.compressed_imgmsg_to_cv2(
                        img_r_raw, desired_encoding="rgb8"
                    )

                    state = np.array(act_raw.position)

                if debug:
                    os.makedirs("frames", exist_ok=True)
                    img_h_pil = Image.fromarray(img_h) # (720, 1280, 3)
                    img_l_pil = Image.fromarray(img_l) # (480, 848, 3)
                    img_r_pil = Image.fromarray(img_r) # (480, 848, 3)

                    img_l_pil.save(f'frames/wrist_l_{count:05d}.png')
                    img_h_pil.save(f'frames/head_{count:05d}.png')
                    img_r_pil.save(f'frames/wrist_r_{count:05d}.png')

                    print(f"Saved frame {count} images.")
                    print(f" - Saved images to path: frames/head_{count:05d}.png, frames/wrist_l_{count:05d}.png, frames/wrist_r_{count:05d}.png")

                    if cfg.with_proprio:
                        action_queue = policy.step(img_h, img_l, img_r, lang, state)
                    else:
                        action_queue = policy.step(img_h, img_l, img_r, lang)

        sim_ros_node.loop_rate.sleep()


@dataclass
class GenerateConfig:

    name = "finetuned"

    model_family: str = "openvla"  # Model family
    pretrained_checkpoint: Union[str, Path] = f"checkpoints/{name}"

    load_in_8bit: bool = False  # (For OpenVLA only) Load with 8-bit quantization
    load_in_4bit: bool = False  # (For OpenVLA only) Load with 4-bit quantization

    center_crop: bool = False  # Center crop? (if trained w/ random crop image aug)
    local_log_dir: str = "./experiments/eval_logs"  # Local directory for eval logs
    seed: int = 0

    action_decoder_path: str = f"checkpoints/{name}/action_decoder.pt"
    window_size: int = 30

    n_layers: int = 2
    hidden_dim: int = 1024

    with_proprio: bool = True
    wogripper: bool = True

    smooth: bool = False
    balancing_factor: float = 0.01  # larger for smoother

    task_name: str = "iros_pack_in_the_supermarket"


@draccus.wrap()
def get_policy(cfg: GenerateConfig) -> None:

    wrapped_model = WrappedModel(cfg)
    wrapped_model.cuda()
    wrapped_model.eval()
    policy = WrappedGenieEvaluation(cfg, wrapped_model)

    return policy, cfg


if __name__ == "__main__":
    policy, cfg = get_policy()
    infer(policy, cfg)