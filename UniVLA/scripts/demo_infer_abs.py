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

    lang = get_instruction(cfg.task_name)

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
             
            if ((init_frame or infer_start) and
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

                    count = count + 1
                    img_h = bridge.compressed_imgmsg_to_cv2(img_h_raw, desired_encoding="rgb8")
                    img_l = bridge.compressed_imgmsg_to_cv2(img_l_raw, desired_encoding="rgb8")
                    img_r = bridge.compressed_imgmsg_to_cv2(img_r_raw, desired_encoding="rgb8")

                    state = np.array(act_raw.position[0:16])

                    # 计算目标位置
                    # target ee pose in base 坐标系
                    target_pose_in_base = np.array([
                        [-9.84044812e-01,  2.01776121e-02,  1.76772936e-01, 0.927342],
                        [ 1.77920481e-01,  1.13445097e-01,  9.77483760e-01, -0.0591086],
                        [-3.30735790e-04,  9.93339349e-01, -1.15225081e-01, 1.0373207],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
                    ])
                    # current gripper_r_center_link in base 坐标系
                    current_right_ee_pose_in_base = np.array([
                        [-9.84044812e-01,  2.01776121e-02,  1.76772936e-01, 6.24140263e-01],
                        [ 1.77920481e-01,  1.13445097e-01,  9.77483760e-01, -1.62116051e-01],
                        [-3.30735790e-04,  9.93339349e-01, -1.15225081e-01, 7.85317540e-01],
                        [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]
                    ])
                    # 计算delta actions在base坐标系下
                    # 左手臂保持不变，右手臂移动到目标位置
                    delta_left = np.zeros(6)  # 左手臂不动

                    # 计算右手臂的相对变换矩阵
                    delta_right_matrix = np.linalg.inv(current_right_ee_pose_in_base) @ target_pose_in_base
                    delta_right = mat2xyzrpy(delta_right_matrix)

                    # delta ee pose | base coordinate
                    abs_actions = [
                        np.concatenate([
                            delta_left,     # left arm: no change (6 elements)
                            delta_right,    # right arm: move to target (6 elements) 
                            [0.0, 0.0]      # grippers: no change (2 elements)
                        ])
                    ]

                    arm_joint_state = np.array(list(state[0:7]) + list(state[8:15]))
                    abs_eef_action = ik_fk_solver.compute_abs_eef_from_base(abs_actions, arm_joint_state) # ee pose in center coordinate system, len(abs_eef_action[0]) = 14
                    joint_actions = ik_fk_solver.eef_actions_to_joint(abs_eef_action, arm_joint_state, init_head) # joint_actions, len(joint_actions[0]) = 16
                    
                   

                    for i, joint_action in enumerate(joint_actions):
                        joint_cmd = []
                        # Fill joint command with arm and gripper commands
                        # Format: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
                        joint_cmd.extend(joint_action[0:7])   # Left arm joints
                        joint_cmd.extend(joint_action[14:15]) # Left gripper  
                        joint_cmd.extend(joint_action[7:14])  # Right arm joints
                        joint_cmd.extend(joint_action[15:16]) # Right gripper
                        pub_msg_buffer.append(joint_cmd)

                else:
                    # init ik fk solver
                    if init_arm is None:
                        # Get initial arm joint positions from ROS topic (sim_ros_node.get_joint_state() )
                        init_arm = []
                        for i in range(7):
                            init_arm.append(act_raw.position[i])
                            init_arm.append(act_raw.position[i + 8])
                        
                        # Get waist and head joints from ROS topic (sim_ros_node.cur_joint_state)
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
                            # 获取两个坐标系在世界坐标系中的位姿
                            # 这里需要你从仿真环境中获取实际的变换矩阵
                            base_link_in_world = np.array([
                                [ 1.00000000e+00,  0.00000000e+00,  0.00000000e+00, -5.10262012e+00],
                                [ 0.00000000e+00,  1.00000000e+00,  0.00000000e+00,  1.10153799e+01],
                                [ 0.00000000e+00,  0.00000000e+00,  1.00000000e+00,  2.23517418e-08],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
                            ])

                            arm_base_link_in_world = np.array([
                                [ 8.74647618e-01, -1.57370774e-09,  4.84759226e-01, -4.82376909e+00],
                                [ 5.61328180e-08,  9.99999980e-01, -9.80336749e-08,  1.10153799e+01],
                                [-4.84759226e-01,  1.12955824e-07,  8.74647618e-01,  9.35265481e-01],
                                [ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]
                            ])
                            
                            # 计算base_T_center
                            base_T_center = None
                            print(f"base_T_center: {base_T_center}")

                            # 使用自定义的base_T_center初始化
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
