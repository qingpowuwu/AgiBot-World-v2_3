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
    
    # Test command flag - 测试命令标志
    test_command_sent = False

    lang = get_instruction(cfg.task_name)

    # Main control loop - 主控制循环
    while rclpy.ok():
        # Phase 1: Command execution phase - 命令执行阶段
        # Process buffered joint commands if any exist
        if pub_msg_buffer:
            # Check if this is the last command in buffer
            # is_end: 标记是否为缓冲区中的最后一个命令
            is_end = True if len(pub_msg_buffer) == 1 else False
            # Send joint command to robot and remove from buffer
            # 发送关节命令到机器人并从缓冲区移除
            sim_ros_node.publish_joint_command(pub_msg_buffer.popleft(), is_end)
        else:
            # Phase 2: Perception and planning phase - 感知与规划阶段
            # No commands in buffer, gather sensor data for next inference
            # 缓冲区无命令，收集传感器数据进行下一次推理
            
            # Gather multi-modal sensor data - 收集多模态传感器数据
            img_h_raw = sim_ros_node.get_img_head()        # Head camera image - 头部相机图像
            img_l_raw = sim_ros_node.get_img_left_wrist()  # Left wrist camera image - 左腕相机图像
            img_r_raw = sim_ros_node.get_img_right_wrist() # Right wrist camera image - 右腕相机图像
            act_raw = sim_ros_node.get_joint_state()       # Current joint state (16 joints) - 当前关节状态
            infer_start = sim_ros_node.is_infer_start()    # Inference trigger signal - 推理触发信号
            
            # Data synchronization check - 数据同步检查
            # Ensure all sensor data is valid and temporally synchronized
            if ((init_frame or infer_start) and
                (
                    img_h_raw
                    and img_l_raw
                    and img_r_raw
                    and act_raw
                    # Temporal synchronization: all images captured at same timestamp
                    # 时间同步：所有图像在相同时间戳捕获
                    and img_h_raw.header.stamp == img_l_raw.header.stamp == img_r_raw.header.stamp
                )
            ):
                # Simulation time check - 仿真时间检查
                sim_time = get_sim_time(sim_ros_node)
                if sim_time > SIM_INIT_TIME and ik_fk_solver is not None:  # Wait for simulation to stabilize - 等待仿真稳定
                    init_frame = False  # Disable initialization flag - 禁用初始化标志

                    count = count + 1
                    
                    # 将压缩的ROS图像转换为OpenCV格式(RGB8)
                    img_h = bridge.compressed_imgmsg_to_cv2(img_h_raw, desired_encoding="rgb8")
                    img_l = bridge.compressed_imgmsg_to_cv2(img_l_raw, desired_encoding="rgb8") 
                    img_r = bridge.compressed_imgmsg_to_cv2(img_r_raw, desired_encoding="rgb8")

                    # Current robot state in joint space (configuration space)
                    # state: 机器人在关节空间的当前状态 (配置空间)
                    # Format: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
                    state = np.array(act_raw.position[0:16])

                    # Test movement: Right arm x+0.1, z+0.1 (execute only once)
                    # 测试运动：右臂 x+0.1, z+0.1 (只执行一次)
                    if not test_command_sent:
                        print("Generating test movement: Right arm x+0.1, z+0.1")
                        
                        # Create test action in EE delta space
                        # test_action: 末端执行器增量空间的测试动作
                        # Format: [left_eef_delta(6), right_eef_delta(6), gripper_actions(2)]
                        test_action = [
                            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # Left arm no movement - 左臂不动
                            0.1, 0.0, 0.1, 0.0, 0.0, 0.0,  # Right arm: x+0.1m, z+0.1m in base frame - 右臂移动
                            0.0, 0.0  # Grippers no action - 夹爪不动
                        ]
                        
                        # abs_actions: 策略输出的绝对动作 in EE space  
                        abs_actions = [test_action]  # Wrap as action list for processing
                    else:
                        # Policy inference - 策略推理 (for future implementation)
                        # payload: 多模态输入数据包 (images + joint state + task context)
                        payload = None  # Should contain: {images, joint_state, task_info}
                        # abs_actions: 策略输出的绝对动作 in EE space
                        # Format: [left_eef_delta(6), right_eef_delta(6), gripper_actions(2)]
                        if policy is not None:
                            abs_actions = policy.infer(payload)
                        else:
                            abs_actions = [[0.0] * 14]  # No movement if no policy

                    # Kinematics processing pipeline - 运动学处理流水线
                    # Step 1: Convert joint format for IK solver input
                    # arm_joint_state: 手臂关节状态，去除夹爪，顺序格式
                    # Format: [left_arm(7), right_arm(7)] - excluding grippers
                    arm_joint_state = np.array(list(state[0:7]) + list(state[8:15]))
                    
                    # Step 2: Forward kinematics + delta transformation
                    # abs_eef_action: 绝对末端执行器目标姿态 in robot base frame
                    # Current EE pose (FK) + policy delta → Target EE pose (absolute)
                    abs_eef_action = ik_fk_solver.compute_abs_eef_from_base(abs_actions, arm_joint_state)
                    
                    # Step 3: Inverse kinematics to joint space
                    # joint_actions: 目标关节角度序列 (trajectory in joint space)
                    # Target EE pose → Target joint angles via IK
                    joint_actions = ik_fk_solver.eef_actions_to_joint(abs_eef_action, arm_joint_state, init_head)
                    
                    # Command generation and buffering - 命令生成与缓冲
                    # Convert joint trajectory to ROS command format
                    for i, joint_action in enumerate(joint_actions):
                        joint_cmd = []
                        # Fill joint command with arm and gripper commands
                        # Format: [left_arm(7), left_gripper(1), right_arm(7), right_gripper(1)]
                        joint_cmd.extend(joint_action[0:7])   # Left arm joints
                        joint_cmd.extend(joint_action[14:15]) # Left gripper  
                        joint_cmd.extend(joint_action[7:14])  # Right arm joints
                        joint_cmd.extend(joint_action[15:16]) # Right gripper
                        
                        # Add to command buffer for execution in next cycles
                        # 添加到命令缓冲区，在下一循环周期执行
                        pub_msg_buffer.append(joint_cmd)
                    
                    # Mark test command as sent - 标记测试命令已发送
                    if not test_command_sent:
                        test_command_sent = True
                        print(f"Added {len(joint_actions)} joint commands to buffer")

                else:
                    # init ik fk solver
                    if init_arm is None:
                        init_arm = []
                        for i in range(7):
                            init_arm.append(act_raw.position[i])      # Left arm joint i
                            init_arm.append(act_raw.position[i + 8])  # Right arm joint i
                        
                        
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
                        
                        print(f"init_arm (interleaved): {init_arm}")
                        print(f"init_waist: {init_waist}")
                        print(f"init_head: {init_head}")
                        
                        if ik_fk_solver is None:
                            # Initialize dual-arm IK/FK solver
                            ik_fk_solver = IKFKSolver(init_arm,  # len(init_arm) == 14
                                                      init_head, # len(init_head) == 2
                                                      init_waist # len(init_waist) == 2
                                                      )
                            print("IK/FK solver initialized successfully")

        # Control loop timing - 控制循环时序
        # Sleep to maintain control frequency (typically 10-100 Hz)
        # 休眠以维持控制频率 (通常10-100Hz)
        sim_ros_node.loop_rate.sleep()


@dataclass
class DeployConfig:
  # To be implemented
  task_name: str = "test_movement"


@draccus.wrap()
def get_policy(cfg: DeployConfig) -> None:
    # To be implemented
    policy = None
    return policy, cfg


if __name__ == "__main__":
    policy, cfg = get_policy()
    infer(policy, cfg)
