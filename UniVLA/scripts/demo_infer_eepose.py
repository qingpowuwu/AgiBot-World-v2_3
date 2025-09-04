import os
import sys
from pathlib import Path
import numpy as np
import time
import json
from dataclasses import dataclass
from typing import List

# 添加路径
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent))

from PIL import Image
from dataclasses import dataclass
from typing import Union

import cv2
import numpy as np
import draccus
# from experiments.robot.geniesim.genie_model import WrappedGenieEvaluation, WrappedModel

import rclpy
import time, threading
from cv_bridge import CvBridge

import rclpy
import threading
from genie_sim_ros import SimROSNode
import source.geniesim.utils.system_utils as system_utils
from source.geniesim.benchmark.envs.dummy_env import DummyEnv
from source.geniesim.robot.genie_robot import IsaacSimRpcRobot
from source.geniesim.layout.task_generate import TaskGenerator
from ikfk_utils import IKFKSolver
from collections import deque

@dataclass
class GraspTestConfig:
    task_name: str = "iros_clear_the_countertop_waste"
    client_host: str = "localhost:50051"
    
    # 测试抓取姿态
    test_grasp_pose: List[List[float]] = None
    
    def __post_init__(self):
        if self.test_grasp_pose is None:
            # 你提供的抓取姿态(在 robo base 坐标系下)
            # self.test_grasp_pose = [
            #     [ 0.994734,  -0.0157556,  0.101268,   0.927342 ],
            #     [-0.0591397, -0.895246,   0.44163,   -0.0591086],
            #     [ 0.0837021, -0.445293,  -0.891464,  -0.0201597],
            #     [ 0.0,        0.0,        0.0,        1.0       ]
            # ]
            # 你提供的抓取姿态(在 robo base 坐标系下 with fixed z)
            self.test_grasp_pose = [
                [ 0.994734,  -0.0157556,  0.101268,   0.927342 ],
                [-0.0591397, -0.895246,   0.44163,   -0.0591086],
                [ 0.0837021, -0.445293,  -0.891464,  0.80],
                [ 0.0,        0.0,        0.0,        1.0       ]
            ]
            # 你提供的抓取姿态(在 world 坐标系下)
            # self.test_grasp_pose = [
            #     [ 0.994734,  -0.0157556,  0.101268,   -4.137342 ],
            #     [-0.0591397, -0.895246,   0.44163,   10.95],
            #     [ 0.0837021, -0.445293,  -0.891464,  0.80],
            #     [ 0.0,        0.0,        0.0,        1.0       ]
            # ]

class GraspExecutor:
    def __init__(self, cfg: GraspTestConfig):
        self.cfg = cfg
        self.sim_ros_node = None
        self.robot = None
        self.env = None
        self.ik_fk_solver = None
        self.init_arm = None
        self.init_head = [0.0, 0.0]
        self.init_waist = None
        self.pub_msg_buffer = deque(maxlen=30)
        
    def create_environment_and_robot(self):
        """创建环境和机器人"""
        print(f"正在为任务 {self.cfg.task_name} 创建环境...")
        
        # 1. 加载任务配置文件
        task_config_file = os.path.join(
            system_utils.benchmark_ader_path(), "eval_tasks", self.cfg.task_name + ".json"
        )
        
        if not os.path.exists(task_config_file):
            raise FileNotFoundError(f"任务配置文件不存在: {task_config_file}")
        
        task_config = system_utils.load_json(task_config_file)
        task_config["specific_task_name"] = self.cfg.task_name
        
        # 2. 生成任务场景配置
        task_generator = TaskGenerator(task_config)
        task_folder = os.path.join(
            system_utils.benchmark_root_path(), 
            "saved_task/%s" % task_config["task"]
        )
        
        # 生成一个任务实例
        task_generator.generate_tasks(
            save_path=task_folder,
            task_num=1,
            task_name=task_config["task"],
        )
        
        # 更新机器人初始位姿
        robot_position = task_generator.robot_init_pose["position"]
        robot_rotation = task_generator.robot_init_pose["quaternion"]
        task_config["robot"]["robot_init_pose"]["position"] = robot_position
        task_config["robot"]["robot_init_pose"]["quaternion"] = robot_rotation
        
        # 3. 获取生成的任务文件
        import glob
        specific_task_files = glob.glob(task_folder + "/*.json")
        if not specific_task_files:
            raise FileNotFoundError(f"没有找到任务文件: {task_folder}")
        
        episode_file = specific_task_files[0]
        
        # 4. 创建机器人
        robot_cfg = task_config["robot"]["robot_cfg"]
        self.robot = IsaacSimRpcRobot(
            robot_cfg=robot_cfg,
            scene_usd=task_config["scene"]["scene_usd"],
            client_host=self.cfg.client_host,
            position=task_config["robot"]["robot_init_pose"]["position"],
            rotation=task_config["robot"]["robot_init_pose"]["quaternion"],
            gripper_control_type=0,  # position control
        )
        
        # 5. 创建环境
        self.env = DummyEnv(self.robot, episode_file, task_config)
        
        # 6. 设置机器人初始姿态
        init_pose = task_config["robot"].get("init_arm_pose")
        if init_pose:
            self.robot.set_init_pose(init_pose)
        
        print("环境和机器人创建完成")
        return task_config
    
    def init_ros(self):
        """初始化ROS节点"""
        if not rclpy.ok():
            rclpy.init()
        self.sim_ros_node = SimROSNode()
        spin_thread = threading.Thread(target=rclpy.spin, args=(self.sim_ros_node,))
        spin_thread.daemon = True
        spin_thread.start()
        
        time.sleep(2.0)
        print("ROS节点初始化完成")
    
    def get_current_joint_state(self):
        """获取当前关节状态"""
        max_retries = 10
        retry_count = 0
        
        while retry_count < max_retries:
            act_raw = self.sim_ros_node.get_joint_state()
            if act_raw and hasattr(act_raw, 'position') and len(act_raw.position) > 0:
                return np.array(act_raw.position)
            
            retry_count += 1
            time.sleep(0.1)
        
        print("无法获取关节状态")
        return None
    
    def init_ik_fk_solver(self):
        """初始化IK/FK求解器，使用与demo_infer_abs.py相同的逻辑"""
        if self.ik_fk_solver is not None:
            return True
            
        # Get waist and head joints from ROS topic (sim_ros_node.cur_joint_state)
        cur_joint_state = self.sim_ros_node.cur_joint_state
        if cur_joint_state is None:
            print("无法获取当前关节状态，稍后重试...")
            return False
            
        joint_name_state_dict = {}
        for idx, name in enumerate(cur_joint_state.name):
            joint_name_state_dict[name] = cur_joint_state.position[idx]
        
        try:
            # Get waist joints (body joints)
            self.init_waist = [
                joint_name_state_dict["idx02_body_joint2"],
                joint_name_state_dict["idx01_body_joint1"]
            ]
            
            # Get head joints
            self.init_head = [
                joint_name_state_dict["idx11_head_joint1"],
                joint_name_state_dict["idx12_head_joint2"]
            ]

            self.init_arm = [
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
            
            # TBD waist, init_arm = [-1.074, 1.075, 0.6106, -0.6114, 0.2808, -0.2807, -1.2838, 1.2838, 0.72, -0.7319, 1.4951, -1.4952, -0.186, 0.1876]
            self.ik_fk_solver = IKFKSolver(self.init_arm, self.init_head, self.init_waist)
            
            print(f"✅ IK/FK求解器初始化成功")
            print(f"初始手臂关节: {self.init_arm}")
            print(f"初始头部关节: {self.init_head}")
            print(f"初始腰部关节: {self.init_waist}")
            return True
            
        except KeyError as e:
            print(f"缺少关节: {e}")
            return False
    
    def matrix_to_quaternion(self, rotation_matrix):
        """将旋转矩阵转换为四元数"""
        from scipy.spatial.transform import Rotation
        r = Rotation.from_matrix(rotation_matrix)
        return r.as_quat()  # 返回 [x, y, z, w] 格式
    
    def pose_matrix_to_delta_ee_pose(self, target_pose_matrix, current_joints):
        """
        将目标姿态矩阵转换为delta_ee_pose格式，使用与demo_infer_abs.py相同的逻辑
        
        Args:
            target_pose_matrix: 4x4目标姿态矩阵 (base坐标系)
            current_joints: 当前关节状态
            
        Returns:
            delta_ee_pose: 格式为 [[dx0, dy0, dz0, dR0, dP0, dY0, dx1, dy1, dz1, dR1, dP1, dY1, eef0, eef1]]
        """
        # 使用IK/FK求解器获取当前右臂末端位姿
        arm_joint_state = np.array(list(current_joints[0:7]) + list(current_joints[8:15]))
        current_ee_poses = self.ik_fk_solver.compute_ee_pose_from_base(arm_joint_state)
        
        # 提取右臂当前末端位姿 (索引1是右臂)
        current_right_ee_pose = current_ee_poses[1]  # shape: (4, 4)
        
        # 计算位置差值
        target_position = target_pose_matrix[:3, 3]
        current_position = current_right_ee_pose[:3, 3]
        delta_position = target_position - current_position
        
        # 计算旋转差值（使用欧拉角）
        from scipy.spatial.transform import Rotation
        current_rotation = Rotation.from_matrix(current_right_ee_pose[:3, :3])
        target_rotation = Rotation.from_matrix(target_pose_matrix[:3, :3])
        
        # 计算相对旋转
        relative_rotation = target_rotation * current_rotation.inv()
        delta_euler = relative_rotation.as_euler('xyz')  # Roll, Pitch, Yaw
        
        # 构造delta_ee_pose (左臂不动，只动右臂)
        delta_ee_pose = [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # 左臂: 不变
             delta_position[0], delta_position[1], delta_position[2],  # 右臂: 位置增量
             delta_euler[0], delta_euler[1], delta_euler[2],  # 右臂: 旋转增量(RPY)
             0.0, 0.0]  # 夹爪: 不变
        ]
        
        return delta_ee_pose
    
    def solve_ik_for_pose(self, target_pose_matrix, current_joints):
        """使用IK/FK求解器计算关节角度，使用与demo_infer_abs.py相同的逻辑"""
        try:
            # 转换为delta_ee_pose格式
            delta_ee_pose = self.pose_matrix_to_delta_ee_pose(target_pose_matrix, current_joints)
            
            print(f"Delta EE pose: {delta_ee_pose[0]}")
            
            # 使用IK/FK求解器计算绝对末端姿态
            arm_joint_state = np.array(list(current_joints[0:7]) + list(current_joints[8:15]))
            abs_eef_action = self.ik_fk_solver.compute_abs_eef_from_base(delta_ee_pose, arm_joint_state)
            
            print(f"绝对末端动作: {abs_eef_action}")
            
            # 计算关节动作
            joint_actions = self.ik_fk_solver.eef_actions_to_joint(abs_eef_action, arm_joint_state, self.init_head)
            
            if joint_actions is not None and len(joint_actions) > 0:
                # 使用第一个解
                joint_action = joint_actions[0]
                
                # 构造完整的关节命令
                new_joints = current_joints.copy()
                
                # 填充关节命令（与demo_infer_abs.py相同的映射）
                joint_cmd = []
                joint_cmd.extend(joint_action[0:7])   # Left arm joints
                joint_cmd.extend(joint_action[14:15]) # Left gripper 
                joint_cmd.extend(joint_action[7:14])  # Right arm joints
                joint_cmd.extend(joint_action[15:16]) # Right gripper
                
                # 更新关节状态
                if len(joint_cmd) >= 16:
                    new_joints[:16] = joint_cmd[:16]
                
                print(f"✅ IK求解成功!")
                print(f"目标关节角度(右臂): {joint_action[7:14]}")
                return new_joints
            else:
                print("❌ IK求解失败")
                return None
                
        except Exception as e:
            print(f"❌ IK求解过程中发生错误: {e}")
            return None
    
    def move_robot_to_pose(self, target_joints):
        """移动机器人到指定关节位置，使用与demo_infer_abs.py相同的发布逻辑"""
        try:
            print(f"发送关节命令: {target_joints[8:15].round(3)}... (右臂关节)")
            
            # 清空缓冲区
            self.pub_msg_buffer.clear()
            
            # 添加关节命令到缓冲区
            self.pub_msg_buffer.append(target_joints.tolist())
            
            # 发布命令（模拟demo_infer_abs.py的发布逻辑）
            max_iterations = 100
            iteration = 0
            
            while self.pub_msg_buffer and iteration < max_iterations:
                is_end = True if len(self.pub_msg_buffer) == 1 else False
                self.sim_ros_node.publish_joint_command(self.pub_msg_buffer.popleft(), is_end)
                
                # 等待一小段时间
                time.sleep(0.1)
                iteration += 1
            
            # 等待运动完成
            time.sleep(4.0)
            
            # 验证是否到达
            current_joints = self.get_current_joint_state()
            if current_joints is not None:
                # 检查右臂关节误差
                right_arm_error = np.linalg.norm(current_joints[8:15] - target_joints[8:15])
                print(f"右臂关节误差: {right_arm_error:.4f}")
                return right_arm_error < 0.3
            return False
        except Exception as e:
            print(f"移动机器人失败: {e}")
            return False
    
    def execute_grasp_pose(self, grasp_pose_matrix):
        """
        执行抓取姿态
        
        Args:
            grasp_pose_matrix: 4x4变换矩阵，表示在机器人base坐标系中的抓取姿态
        
        Returns:
            bool: 是否成功执行抓取
        """
        try:
            print("=" * 50)
            print("开始执行抓取姿态测试...")
            print("=" * 50)
            
            # 初始化IK/FK求解器
            if not self.init_ik_fk_solver():
                print("❌ IK/FK求解器初始化失败")
                return False
            
            grasp_pose = np.array(grasp_pose_matrix)
            print(f"抓取姿态矩阵:\n{grasp_pose}")
            print(f"抓取位置(base坐标系): [{grasp_pose[0,3]:.3f}, {grasp_pose[1,3]:.3f}, {grasp_pose[2,3]:.3f}]")
            
            # 获取当前关节状态作为IK求解的初始值
            current_joints = self.get_current_joint_state()
            if current_joints is None:
                print("❌ 无法获取当前关节状态")
                return False
            
            print(f"当前关节状态(右臂): {current_joints[8:15].round(3)}")
            
            # 使用IK求解目标关节角度
            print("正在求解IK...")
            target_joints = self.solve_ik_for_pose(grasp_pose, current_joints)
            
            if target_joints is None:
                print("❌ IK求解失败，无法到达目标抓取姿态")
                print("可能的原因:")
                print("  1. 目标位置超出机器人工作空间")
                print("  2. 目标姿态不可达")
                print("  3. 存在奇异性问题")
                return False
            
            # 移动机器人到目标位置
            print("开始移动机器人...")
            success = self.move_robot_to_pose(target_joints)
            
            if success:
                print("✅ 机器人成功移动到抓取姿态!")
                
                # 验证最终位置（如果需要的话）
                print("验证最终位置...")
                time.sleep(1.0)
                
                return True
            else:
                print("❌ 机器人移动失败")
                return False
                
        except Exception as e:
            print(f"❌ 执行抓取姿态时发生错误: {e}")
            return False
    
    def execute_grasp_sequence(self, grasp_pose_matrix, approach_distance=0.1):
        """
        执行完整的抓取序列：接近 -> 抓取 -> 后退
        
        Args:
            grasp_pose_matrix: 抓取姿态 (4x4矩阵)
            approach_distance: 接近距离
        
        Returns:
            bool: 是否成功执行完整序列
        """
        try:
            print("=" * 60)
            print("开始完整抓取序列测试...")
            print("=" * 60)
            
            grasp_pose = np.array(grasp_pose_matrix)
            
            # 1. 接近阶段
            print("\n--- 阶段1: 接近目标 ---")
            approach_pose = grasp_pose.copy()
            # 沿着抓取姿态的-Z方向后退
            approach_offset = -approach_distance * grasp_pose[:3, 2]
            approach_pose[:3, 3] += approach_offset
            
            print(f"接近位置: [{approach_pose[0,3]:.3f}, {approach_pose[1,3]:.3f}, {approach_pose[2,3]:.3f}]")
            
            if not self.execute_grasp_pose(approach_pose):
                print("❌ 无法到达接近位置")
                return False
            
            print("✅ 成功到达接近位置!")
            time.sleep(1.0)
            
            # 2. 抓取阶段
            print("\n--- 阶段2: 执行抓取 ---")
            print(f"最终抓取位置: [{grasp_pose[0,3]:.3f}, {grasp_pose[1,3]:.3f}, {grasp_pose[2,3]:.3f}]")
            
            if not self.execute_grasp_pose(grasp_pose):
                print("❌ 无法到达最终抓取位置")
                return False
            
            print("✅ 成功到达抓取位置!")
            
            # 这里可以添加夹爪控制
            print("模拟夹爪抓取动作...")
            time.sleep(2.0)
            
            # 3. 后退阶段
            print("\n--- 阶段3: 后退 ---")
            retreat_pose = grasp_pose.copy()
            retreat_offset = -approach_distance * grasp_pose[:3, 2]
            retreat_pose[:3, 3] += retreat_offset
            
            print(f"后退位置: [{retreat_pose[0,3]:.3f}, {retreat_pose[1,3]:.3f}, {retreat_pose[2,3]:.3f}]")
            
            if not self.execute_grasp_pose(retreat_pose):
                print("❌ 无法完成后退")
                return False
            
            print("✅ 成功完成后退!")
            print("=" * 60)
            print("🎉 抓取序列完成！")
            print("=" * 60)
            return True
            
        except Exception as e:
            print(f"❌ 执行抓取序列时发生错误: {e}")
            return False
    
    def run_test(self):
        """运行测试"""
        try:
            # 1. 创建环境
            print("步骤1: 创建环境和机器人...")
            task_config = self.create_environment_and_robot()
            
            # 2. 初始化ROS
            print("步骤2: 初始化ROS...")
            self.init_ros()
            
            # 3. 等待系统稳定
            print("步骤3: 等待系统初始化...")
            time.sleep(5.0)
            
            # 4. 重置环境
            print("步骤4: 重置环境...")
            observation = self.env.reset()
            time.sleep(2.0)
            
            # 5. 等待更长时间以确保IK/FK求解器可以正确初始化
            print("步骤5: 等待IK/FK求解器就绪...")
            max_wait_time = 30  # 最多等待30秒
            wait_time = 0
            while wait_time < max_wait_time:
                if self.init_ik_fk_solver():
                    break
                time.sleep(1.0)
                wait_time += 1
                print(f"等待IK/FK求解器初始化... {wait_time}/{max_wait_time}")
            
            if self.ik_fk_solver is None:
                print("❌ IK/FK求解器初始化超时")
                return
            
            # 6. 测试抓取姿态
            print("步骤6: 开始抓取测试...")
            
            # 选择测试模式
            test_mode = input("\n选择测试模式:\n1. 直接抓取\n2. 完整抓取序列(接近->抓取->后退)\n请输入(1或2): ").strip()
            
            if test_mode == "1":
                success = self.execute_grasp_pose(self.cfg.test_grasp_pose)
            elif test_mode == "2":
                success = self.execute_grasp_sequence(self.cfg.test_grasp_pose, approach_distance=0.08)
            else:
                print("默认使用完整抓取序列...")
                success = self.execute_grasp_sequence(self.cfg.test_grasp_pose, approach_distance=0.08)
            
            if success:
                print("\n🎉 测试成功完成!")
            else:
                print("\n❌ 测试失败!")
            
            # 7. 清理资源
            print("步骤7: 清理资源...")
            self.robot.client.Exit()
            
        except Exception as e:
            print(f"❌ 测试过程中发生错误: {e}")
            if self.robot:
                self.robot.client.Exit()

def main():
    """主函数"""
    cfg = GraspTestConfig(
        task_name="iros_clear_the_countertop_waste",
        client_host="localhost:50051"
    )
    
    print("抓取执行测试程序")
    print("=" * 50)
    print(f"任务: {cfg.task_name}")
    print(f"测试抓取姿态:")
    for row in cfg.test_grasp_pose:
        print(f"  {row}")
    print("=" * 50)
    
    executor = GraspExecutor(cfg)
    executor.run_test()

if __name__ == "__main__":
    main()
