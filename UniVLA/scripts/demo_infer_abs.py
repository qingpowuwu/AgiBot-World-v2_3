import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

import rclpy
import time, threading
from cv_bridge import CvBridge
import cv2
from genie_sim_ros import SimROSNode

import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import draccus
from PIL import Image
import torch
from ikfk_utils import IKFKSolver, cal_base_T_center, mat2xyzrpy
import ik_solver
import itertools
from collections import deque


def get_sim_time(sim_ros_node):
    sim_time = sim_ros_node.get_clock().now().nanoseconds * 1e-9
    return sim_time

def get_instruction(task_name):
    pass

def infer(policy, cfg):
    rclpy.init()
    sim_ros_node = SimROSNode()
    spin_thread = threading.Thread(target=rclpy.spin, args=(sim_ros_node,))
    spin_thread.start()
    init_arm = None
    ik_fk_solver = None
    init_head, init_waist = [0.0, 0.0], None
    init_frame = True
    bridge = CvBridge()
    count = 0
    SIM_INIT_TIME = 10
    pub_msg_buffer = deque(maxlen=30)

    task_completed = False
    position_threshold = 0.02  # 降低到2cm
    consecutive_success_count = 0
    required_consecutive_success = 3  # 减少到3次
    last_position = None
    
    while rclpy.ok():
        if pub_msg_buffer:
            is_end = True if len(pub_msg_buffer) == 1 else False
            sim_ros_node.publish_joint_command(pub_msg_buffer.popleft(), is_end)
        else:
            img_h_raw = sim_ros_node.get_img_head()
            img_l_raw = sim_ros_node.get_img_left_wrist()
            img_r_raw = sim_ros_node.get_img_right_wrist()
            act_raw = sim_ros_node.get_joint_state()
            infer_start = sim_ros_node.is_infer_start()
             
            if not task_completed and ((init_frame or infer_start) and
                (
                    img_h_raw
                    and img_l_raw
                    and img_r_raw
                    and act_raw
                    and img_h_raw.header.stamp == img_l_raw.header.stamp == img_r_raw.header.stamp
                )
            ):
                sim_time = get_sim_time(sim_ros_node)
                if sim_time > SIM_INIT_TIME and ik_fk_solver is not None:
                    init_frame = False

                    # 如果还有未执行的动作，等待执行完成
                    if len(pub_msg_buffer) > 5:  # 保留一些缓冲
                        continue

                    count = count + 1
                    img_h = bridge.compressed_imgmsg_to_cv2(img_h_raw, desired_encoding="rgb8")
                    img_l = bridge.compressed_imgmsg_to_cv2(img_l_raw, desired_encoding="rgb8")
                    img_r = bridge.compressed_imgmsg_to_cv2(img_r_raw, desired_encoding="rgb8")

                    state = np.array(act_raw.position[0:16])

                    # 目标位置
                    target_position = np.array([0.927342, -0.0591086, 1.0373207])
                    
                    # 计算当前右手位置 - 统一使用一种方法
                    right_arm_joint_states = state[8:15]
                    
                    # 使用 from_base=True 直接得到在 base_link 坐标系中的位姿
                    current_right_ee_pose_in_base = ik_fk_solver._solver.compute_part_fk(
                        q_part=np.array(right_arm_joint_states, dtype=np.float32),
                        part=ik_solver.RobotPart.RIGHT_ARM,
                        from_base=True,  # 直接计算在 base_link 坐标系中的位姿
                    )

                    # 计算位置误差
                    current_position = current_right_ee_pose_in_base[:3, 3]
                    position_error = np.linalg.norm(target_position - current_position)
                    
                    # 检查位置是否有显著变化
                    if last_position is not None:
                        position_change = np.linalg.norm(current_position - last_position)
                        if position_change < 0.001:  # 如果位置变化很小，跳过这次计算
                            continue
                    
                    last_position = current_position.copy()
                    
                    # 只在位置有明显变化时打印
                    if count % 10 == 0:  # 每10步打印一次
                        print(f"步骤 {count}: 当前位置误差: {position_error:.4f}")
                        print(f"当前位置: {current_position}")
                        print(f"目标位置: {target_position}")
                    
                    # 检查是否达到目标
                    if position_error < position_threshold:
                        consecutive_success_count += 1
                        print(f"达到阈值！连续成功次数: {consecutive_success_count}")
                        if consecutive_success_count >= required_consecutive_success:
                            task_completed = True
                            print("任务完成！")
                            continue
                    else:
                        consecutive_success_count = 0

                    # 动态调整步长
                    if position_error > 0.1:
                        max_position_step = 0.03
                    elif position_error > 0.05:
                        max_position_step = 0.02
                    else:
                        max_position_step = 0.01

                    # 计算移动方向和距离
                    position_direction = target_position - current_position
                    position_distance = np.linalg.norm(position_direction)
                    
                    # 限制位置步长
                    if position_distance > max_position_step:
                        position_direction = position_direction / position_distance * max_position_step
                    
                    # 构建增量动作
                    delta_left = np.zeros(6)
                    delta_right = np.zeros(6)
                    delta_right[:3] = position_direction
                    
                    abs_actions = [
                        np.concatenate([
                            delta_left,     # left arm: no change (6 elements)
                            delta_right,    # right arm: small position step (6 elements) 
                            [0.0, 0.0]      # grippers: no change (2 elements)
                        ])
                    ]

                    try:
                        arm_joint_state = np.array(list(state[0:7]) + list(state[8:15]))
                        abs_eef_action = ik_fk_solver.compute_abs_eef_from_base(abs_actions, arm_joint_state)
                        joint_actions = ik_fk_solver.eef_actions_to_joint(abs_eef_action, arm_joint_state, init_head)

                        for i, joint_action in enumerate(joint_actions):
                            joint_cmd = []
                            joint_cmd.extend(joint_action[0:7])   # Left arm joints
                            joint_cmd.extend(joint_action[14:15]) # Left gripper  
                            joint_cmd.extend(joint_action[7:14])  # Right arm joints
                            joint_cmd.extend(joint_action[15:16]) # Right gripper
                            pub_msg_buffer.append(joint_cmd)
                            
                    except Exception as e:
                        print(f"IK求解失败: {e}")
                        # 如果IK求解失败，跳过这一步
                        continue

                else:
                    # 初始化 IK FK solver
                    if init_arm is None:
                        # Get initial arm joint positions from ROS topic
                        init_arm = []
                        for i in range(7):
                            init_arm.append(act_raw.position[i])
                            init_arm.append(act_raw.position[i + 8])
                        
                        # Get waist and head joints from ROS topic
                        cur_joint_state = sim_ros_node.cur_joint_state
                        joint_name_state_dict = {}
                        for idx, name in enumerate(cur_joint_state.name):
                            joint_name_state_dict[name] = cur_joint_state.position[idx]
                        
                        # Get waist joints (body joints)
                        init_waist = [
                            joint_name_state_dict["idx01_body_joint1"],
                            joint_name_state_dict["idx02_body_joint2"]
                        ]
                        
                        # Get head joints
                        init_head = [
                            joint_name_state_dict["idx11_head_joint1"],
                            joint_name_state_dict["idx12_head_joint2"]
                        ]
                        
                        if ik_fk_solver is None:
                            # 使用你提供的 base_T_center
                            base_T_center = np.array([
                                [ 8.74647618e-01, -1.57370774e-09,  4.84759226e-01,  2.78851030e-01],
                                [ 5.61328180e-08,  9.99999980e-01, -9.80336749e-08,  0.00000000e+00],
                                [-4.84759226e-01,  1.12955824e-07,  8.74647618e-01,  9.35265459e-01],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
                            ])
                            
                            print(f"使用 base_T_center:\n{base_T_center}")
                            ik_fk_solver = IKFKSolver(init_arm, init_head, init_waist, base_T_center=base_T_center)

        sim_ros_node.loop_rate.sleep()

@dataclass
class DeployConfig:
  # Simple test config
  task_name: str = "test_task"



@draccus.wrap()
def get_policy(cfg: DeployConfig) -> None:
    # Simple test policy - returns None since we're hardcoding the action
    policy = None
    return policy, cfg


if __name__ == "__main__":
    policy, cfg = get_policy()
    infer(policy, cfg)
