# uav_manager.py
"""无人机初始化+相机朝向设置"""
import math
import time
from typing import Tuple, Optional
import airsim
from utils import log
import numpy as np

__all__ = [
    "setup_uav",
    "set_camera",
]


def setup_uav(client: airsim.MultirotorClient, name: str, abs_position: Tuple[float, float, float], yaw_deg: float = 0):
    """
    【增强版混合方法】设置无人机位置：先瞬移到目标点附近，再精确飞到目标点。
    包含位置验证和重试机制，以确保无人机正确定位。
    """
    target_x, target_y, target_z = abs_position
    max_retries = 3
    position_tolerance = 0.2  # 20厘米的位置误差容忍

    print(f"\n[DEBUG] === Setting up {name} via Enhanced Hybrid Method ===")

    # 确保控制权并解锁，尝试多次以确保成功
    for i in range(3):
        try:
            client.enableApiControl(True, name)
            client.armDisarm(True, name)
            break
        except Exception as e:
            print(f"[DEBUG] {name} → Control enable attempt {i + 1} failed: {e}")
            time.sleep(0.5)

    # 1) 起飞 - 增加重试机制
    print(f"[DEBUG] {name} → Taking off...")
    for i in range(3):
        try:
            takeoff_result = client.takeoffAsync(vehicle_name=name)
            takeoff_result.join()
            print(f"[DEBUG] {name} → Takeoff complete.")
            break
        except Exception as e:
            print(f"[DEBUG] {name} → Takeoff attempt {i + 1} failed: {e}")
            if i == 2:  # 最后一次尝试失败
                print(f"[DEBUG] {name} → Continuing despite takeoff failures")

    # 等待更长时间以确保起飞稳定
    print(f"[DEBUG] {name} → Stabilization pause after takeoff (1.0s)...")
    time.sleep(1.0)

    # 2) 瞬移到中间点 (目标点上方)
    teleport_offset_z = 2.0
    intermediate_z = target_z - teleport_offset_z
    intermediate_pos = airsim.Vector3r(target_x, target_y, intermediate_z)
    intermediate_quat = airsim.to_quaternion(0, 0, math.radians(yaw_deg))
    intermediate_pose = airsim.Pose(intermediate_pos, intermediate_quat)

    print(
        f"[DEBUG] {name} → Teleporting to intermediate position ({intermediate_pos.x_val:.2f}, {intermediate_pos.y_val:.2f}, {intermediate_pos.z_val:.2f})...")
    for i in range(3):
        try:
            client.simSetVehiclePose(intermediate_pose, ignore_collision=True, vehicle_name=name)
            break
        except Exception as e:
            print(f"[DEBUG] {name} → Teleport attempt {i + 1} failed: {e}")

    # 瞬移后稍长等待，确保位置稳定
    print(f"[DEBUG] {name} → Pause after teleport (0.5s)...")
    time.sleep(0.5)

    # 3) 从中间点精确飞到最终目标，加入重试机制
    final_target_pos = airsim.Vector3r(target_x, target_y, target_z)
    final_target_quat = airsim.to_quaternion(0, 0, math.radians(yaw_deg))
    final_target_pose = airsim.Pose(final_target_pos, final_target_quat)

    fly_velocity = 2.0
    success = False

    for attempt in range(max_retries):
        print(
            f"[DEBUG] {name} → Flying to final target ({target_x:.2f}, {target_y:.2f}, {target_z:.2f}) at {fly_velocity} m/s... (attempt {attempt + 1}/{max_retries})")

        try:
            # 用更长的超时时间
            move_result = client.moveToPositionAsync(
                target_x, target_y, target_z,
                velocity=fly_velocity,
                timeout_sec=10,
                vehicle_name=name,
                yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=yaw_deg)
            )
            move_result.join()
        except Exception as e:
            print(f"[DEBUG] {name} → Move attempt {attempt + 1} failed: {e}")
            time.sleep(0.5)
            continue

        print(f"[DEBUG] {name} → Move command completed, verifying position...")
        time.sleep(1.0)  # 等待位置稳定

        # 验证位置
        current_pose = client.simGetVehiclePose(vehicle_name=name)
        current_pos = current_pose.position.to_numpy_array()
        target_pos_np = np.array([target_x, target_y, target_z])
        diff = current_pos - target_pos_np
        diff_norm = np.linalg.norm(diff)

        print(
            f"[DEBUG] {name} → Position verification: current={current_pos}, target={target_pos_np}, error={diff_norm:.4f}m")

        # 如果位置足够接近目标，则成功
        if diff_norm < position_tolerance:
            print(f"[DEBUG] {name} → Position verified successfully!")
            success = True
            break

    # 4) 如果常规移动方法未能达到目标，尝试直接设置位置
    if not success:
        print(f"[DEBUG] {name} → Regular movement failed, forcing position...")
        try:
            # 强制设置位置
            client.simSetVehiclePose(final_target_pose, ignore_collision=True, vehicle_name=name)
            time.sleep(1.0)  # 等待稳定

            # 再次验证位置
            current_pose = client.simGetVehiclePose(vehicle_name=name)
            current_pos = current_pose.position.to_numpy_array()
            target_pos_np = np.array([target_x, target_y, target_z])
            diff = current_pos - target_pos_np
            print(f"[DEBUG] {name} → After forced positioning: error={np.linalg.norm(diff):.4f}m")
        except Exception as e:
            print(f"[DEBUG] {name} → Force positioning failed: {e}")

    # 5) 命令悬停并等待
    print(f"[DEBUG] {name} → Commanding hover at final position...")
    try:
        client.hoverAsync(vehicle_name=name).join()
    except Exception as e:
        print(f"[DEBUG] {name} → Hover command failed: {e}")

    # 等待更长时间让无人机真正稳定下来
    print(f"[DEBUG] {name} → Final stabilization pause (2.0s)...")
    time.sleep(2.0)

    # 6) 打印最终位置和姿态
    try:
        final_pose_airsim = client.simGetVehiclePose(vehicle_name=name)
        final_pos_np = final_pose_airsim.position.to_numpy_array()
        target_pos_np = np.array(abs_position)
        diff = final_pos_np - target_pos_np
        final_orient_eul = airsim.to_eularian_angles(final_pose_airsim.orientation)

        print(f"[DEBUG] {name} final position: {final_pos_np}")
        print(f"[DEBUG] {name} target position: {target_pos_np}")
        print(f"[DEBUG] {name} difference (Pos): {diff} (norm: {np.linalg.norm(diff):.4f}m)")
        print(f"[DEBUG] {name} difference Z: {diff[2]:.4f}m")

        print(
            f"[DEBUG] {name} final orientation (deg): Pitch={math.degrees(final_orient_eul[0]):.2f}, Roll={math.degrees(final_orient_eul[1]):.2f}, Yaw={math.degrees(final_orient_eul[2]):.2f}")
        print(f"[DEBUG] {name} target orientation (deg): Pitch=0.00, Roll=0.00, Yaw={yaw_deg:.2f}")
    except Exception as e:
        print(f"[DEBUG] {name} → Final status check failed: {e}")

    print(f"[DEBUG] === {name} setup complete ===")


def set_camera(
        client: airsim.MultirotorClient,
        name: str,
        mode: str = "lookdown",
        pitch_deg: Optional[float] = None,
        yaw_deg: Optional[float] = None,
):
    """
    设置主相机的朝向
    :param client:
    :param name:
    :param mode: 三种模式，"lookdown"代表垂直俯视，"horizon"代表平视，"oblique"代表斜视（默认45度，可以手动修改）
    :param pitch_deg: 俯仰角（向下为负）
    :param yaw_deg: 偏航角（机头方向）
    :return:
    """
    if mode == "lookdown":
        pitch_deg = -90
    elif mode == "horizon":
        pitch_deg = 0
    elif mode == "oblique":
        pitch_deg = pitch_deg if pitch_deg is not None else -45
    else:
        raise ValueError("Unknown camera mode")

    yaw_deg = yaw_deg or 0

    # 增加重试机制以确保相机设置成功
    max_retries = 3
    for attempt in range(max_retries):
        try:
            quat = airsim.to_quaternion(math.radians(pitch_deg), 0, math.radians(yaw_deg))
            client.simSetCameraPose("0", airsim.Pose(airsim.Vector3r(0, 0, 0), quat), vehicle_name=name)
            log(f"{name}: camera set → mode={mode} pitch={pitch_deg} yaw={yaw_deg}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                log(f"{name}: camera setting attempt {attempt + 1} failed: {e}, retrying...", "WARN")
                time.sleep(0.2)
            else:
                log(f"{name}: all camera setting attempts failed: {e}", "ERROR")