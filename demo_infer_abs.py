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
from ikfk_utils import IKFKSolver
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

                    # To be implemented
                    payload = None
                    # delta_ee_pose in base_link coordinate
                    # shape 1x1x14: [[dx0, dy0, dz0, dR0, dP0, dY0, dx1, dy1, dz1, dR1, dP1, dY1, eef0, eef1]]
                    delta_ee_pose = policy.infer(payload)  

                    arm_joint_state = np.array(list(state[0:7]) + list(state[8:15]))
                    abs_eef_action = ik_fk_solver.compute_abs_eef_from_base(delta_ee_pose, arm_joint_state)
                    joint_actions = ik_fk_solver.eef_actions_to_joint(abs_eef_action, arm_joint_state, init_head)
                   
                
                    for i, _ in enumerate(joint_actions):
                        joint_cmd = []
                        # To be implemented
                        # - fill joint cmd with arm and gripper cmd
                        pub_msg_buffer.append(joint_cmd)

                else:
                    if init_arm is None and sim_time > SIM_INIT_TIME:
                        cur_joint_state = sim_ros_node.cur_joint_state
                        joint_name_state_dict = {}
                        for idx, name in enumerate(cur_joint_state.name):
                            joint_name_state_dict[name] = cur_joint_state.position[idx]
                        
                        init_waist = [
                            joint_name_state_dict["idx02_body_joint2"],
                            joint_name_state_dict["idx01_body_joint1"]
                        ]
                        
                        init_head = [
                            joint_name_state_dict["idx11_head_joint1"],
                            joint_name_state_dict["idx12_head_joint2"]
                        ]

                        init_arm = [
                            joint_name_state_dict["idx21_arm_l_joint1"],
                            joint_name_state_dict["idx22_arm_l_joint2"],
                            joint_name_state_dict["idx23_arm_l_joint3"],
                            joint_name_state_dict["idx24_arm_l_joint4"],
                            joint_name_state_dict["idx25_arm_l_joint5"],
                            joint_name_state_dict["idx26_arm_l_joint6"],
                            joint_name_state_dict["idx27_arm_l_joint7"],
                            joint_name_state_dict["idx61_arm_r_joint1"],
                            joint_name_state_dict["idx62_arm_r_joint2"],
                            joint_name_state_dict["idx63_arm_r_joint3"],
                            joint_name_state_dict["idx64_arm_r_joint4"],
                            joint_name_state_dict["idx65_arm_r_joint5"],
                            joint_name_state_dict["idx66_arm_r_joint6"],
                            joint_name_state_dict["idx67_arm_r_joint7"]
                        ]
                        
                        if ik_fk_solver is None:
                            ik_fk_solver = IKFKSolver(init_arm, init_head, init_waist) 

        sim_ros_node.loop_rate.sleep()


@dataclass
class DeployConfig:
  # To be implemented
  pass



@draccus.wrap()
def get_policy(cfg: DeployConfig) -> None:
    # To be implemented
    policy = None
    return policy, cfg


if __name__ == "__main__":
    policy, cfg = get_policy()
    infer(policy, cfg)
