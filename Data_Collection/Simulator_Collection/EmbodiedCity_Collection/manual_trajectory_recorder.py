import airsim
import pygame
import os
import csv
import math
import yaml
import time
from datetime import datetime
from multiprocessing import Process, Queue
import numpy as np

def get_xy_offset(client, scene, ref_name="UAV1"):
    """用已飞到位的 ref_name 计算 world 坐标 ↔ scene 坐标 的 XY 偏移"""
    pose = client.simGetVehiclePose(ref_name)
    real_xy = np.array([pose.position.x_val, pose.position.y_val])
    yaml_center = np.array(scene["center"][:2])
    rel_xy = np.array([p["pose"][:2] for p in scene["observer_drones"] if p["name"]==ref_name][0])
    expected_xy = yaml_center + rel_xy
    return real_xy - expected_xy              # Δx,Δy


# ------------------ 通用工具 ------------------

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def get_pose_filename(vehicle_name, pose_dir, tag=""):
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f"{vehicle_name}_{tag}_" if tag else f"{vehicle_name}_"
    return os.path.join(pose_dir, f"{prefix}{ts}.csv")

def pose_writer_thread(pose_queue, pose_path):
    with open(pose_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "x", "y", "z", "pitch", "roll", "yaw"])
        while True:
            pose = pose_queue.get()
            if pose is None:
                break
            w.writerow(pose)
    print(f"[INFO] Pose saved → {pose_path}")

# ------------------ 观察无人机初始化 ------------------

# def setup_observer_uavs(scene):
#     """起飞观察无人机，并精确移动到scene定义的位置"""
#     client = airsim.MultirotorClient()
#     client.confirmConnection()
#
#     center = scene.get("center", [0, 0, 0])
#
#     for obs in scene.get("observer_drones", []):
#         name = obs["name"]
#         rel_pose = obs["pose"]
#         abs_pos = [center[i] + rel_pose[i] for i in range(3)]
#
#         print(f"\n[DEBUG] === Setting up {name} ===")
#         client.enableApiControl(True, name)
#         client.armDisarm(True, name)
#
#         print(f"[DEBUG] {name} → taking off...")
#         client.takeoffAsync(vehicle_name=name).join()
#         print(f"[DEBUG] {name} takeoff complete.")
#
#         time.sleep(1.0)  # 稍等起飞稳定
#
#         # 1. 移动到正确XY
#         cur_pose = client.simGetVehiclePose(vehicle_name=name)
#         cur_z = cur_pose.position.z_val
#
#         print(f"[DEBUG] {name} → moving to target XY ({abs_pos[0]:.2f}, {abs_pos[1]:.2f}) at Z={cur_z:.2f}...")
#         client.moveToPositionAsync(abs_pos[0], abs_pos[1], cur_z, 3, vehicle_name=name).join()
#         time.sleep(0.5)  # 等待稳定
#
#         # 2. 小步调整到目标Z
#         target_z = abs_pos[2]
#         step = -1.0 if target_z < cur_z else 1.0
#         while abs(cur_z - target_z) > 0.5:
#             next_z = cur_z + step
#             if (step < 0 and next_z < target_z) or (step > 0 and next_z > target_z):
#                 next_z = target_z
#             client.moveToZAsync(next_z, 1.0, vehicle_name=name).join()
#             time.sleep(0.2)
#             cur_z = client.simGetVehiclePose(vehicle_name=name).position.z_val
#             print(f"[DEBUG] {name} → moved to Z={cur_z:.2f}m")
#
#         # 3. 设置相机朝向
#         cam_cfg = obs.get("camera", {})
#         mode = cam_cfg.get("mode", "lookdown")
#         if mode == "lookdown":
#             pitch = -90
#         elif mode == "horizon":
#             pitch = 0
#         elif mode == "oblique":
#             pitch = cam_cfg.get("pitch_deg", -45)
#         else:
#             pitch = -90
#         yaw_cam = cam_cfg.get("yaw_deg", 0)
#         cam_quat = airsim.to_quaternion(math.radians(pitch), 0, math.radians(yaw_cam))
#         client.simSetCameraPose("0",
#             airsim.Pose(airsim.Vector3r(0, 0, 0), cam_quat),
#             vehicle_name=name
#         )
#
#         client.hoverAsync(vehicle_name=name).join()
#
#         # 4. 打印最终位置
#         final_pose = client.simGetVehiclePose(vehicle_name=name)
#         final_pos = final_pose.position.to_numpy_array()
#         print(f"[DEBUG] {name} final position: {final_pos}")

# def setup_observer_uavs(scene, client):
#     """
#     起飞观察无人机。先瞬移到目标点，然后用 moveToPositionAsync 精确调整。
#     """
#     center = scene["center"]
#
#     for obs in scene.get("observer_drones", []):
#         name      = obs["name"]
#         rel_pose  = obs["pose"]            # [dx, dy, dz]
#         target_x  = center[0] + rel_pose[0]
#         target_y  = center[1] + rel_pose[1]
#         target_z  = center[2] + rel_pose[2]
#
#         print(f"\n[DEBUG] === Setting up {name} ===")
#         client.enableApiControl(True, name)
#         client.armDisarm(True, name)
#
#         # 1) 起飞并稍微稳定
#         print(f"[DEBUG] {name} → Taking off...")
#         client.takeoffAsync(vehicle_name=name).join()
#         # 添加短暂悬停稳定
#         client.hoverAsync(vehicle_name=name).join()
#         time.sleep(1.0) # 起飞后稳定时间加长一点
#
#         # 2) 直接瞬移到目标 (x, y, z) 和姿态
#         cam_cfg = obs.get("camera", {})
#         pitch   = cam_cfg.get("pitch_deg", -90)
#         yaw     = cam_cfg.get("yaw_deg", 0)
#
#         pose_exact = airsim.Pose(
#             airsim.Vector3r(target_x, target_y, target_z),
#             airsim.to_quaternion(math.radians(pitch), 0, math.radians(yaw))
#         )
#         print(f"[DEBUG] {name} → Teleporting to {pose_exact.position}...")
#         client.simSetVehiclePose(pose_exact, ignore_collision=True, vehicle_name=name)
#         time.sleep(0.1) # 短暂等待瞬移生效
#
#         # 3) 立刻发 0 速度指令清除残余动量
#         print(f"[DEBUG] {name} → Clearing velocity...")
#         client.moveByVelocityAsync(
#             0, 0, 0, duration=0.5,
#             drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
#             yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0),
#             vehicle_name=name
#         ).join()
#         time.sleep(0.1)
#
#         # 4) 使用 moveToPositionAsync 让飞控主动飞到精确目标点
#         #    瞬移后无人机应该已经在目标点附近，这个移动会很快完成
#         print(f"[DEBUG] {name} → Fine-tuning position with moveToPositionAsync to ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})...")
#         # 使用较慢的速度进行微调，例如 1 m/s
#         client.moveToPositionAsync(target_x, target_y, target_z, 1.0, vehicle_name=name).join()
#         time.sleep(0.5) # 等待移动稳定
#
#         # 5) 命令悬停
#         print(f"[DEBUG] {name} → Commanding hover...")
#         client.hoverAsync(vehicle_name=name).join()
#         time.sleep(0.5) # 给悬停命令一点时间稳定
#
#         # 打印最终位置验证
#         final_pose = client.simGetVehiclePose(vehicle_name=name)
#         final_pos  = final_pose.position.to_numpy_array()
#         target_pos = np.array([target_x, target_y, target_z])
#         diff = final_pos - target_pos
#         print(f"[DEBUG] {name} final position: {final_pos}")
#         print(f"[DEBUG] {name} target position: {target_pos}")
#         print(f"[DEBUG] {name} difference: {diff} (norm: {np.linalg.norm(diff):.4f})")

def setup_observer_uavs(scene, client):
    """
    起飞观察无人机。使用最简化设置：起飞 -> 瞬移 -> 悬停。
    模仿 UAV0 的初始定位逻辑，期望能与红点对齐。
    """
    center = scene["center"]

    for obs in scene.get("observer_drones", []):
        name      = obs["name"]
        rel_pose  = obs["pose"]            # [dx, dy, dz]
        target_x  = center[0] + rel_pose[0]
        target_y  = center[1] + rel_pose[1]
        target_z  = center[2] + rel_pose[2]

        print(f"\n[DEBUG] === Setting up {name} ===")
        client.enableApiControl(True, name)
        client.armDisarm(True, name)

        # 1) 起飞
        print(f"[DEBUG] {name} → Taking off...")
        client.takeoffAsync(vehicle_name=name).join()
        # 不再需要起飞后的额外稳定或等待，直接进行下一步

        # 2) 直接瞬移到目标 (x, y, z) 和姿态
        cam_cfg = obs.get("camera", {})
        pitch   = cam_cfg.get("pitch_deg", -90)
        yaw     = cam_cfg.get("yaw_deg", 0)

        pose_exact = airsim.Pose(
            airsim.Vector3r(target_x, target_y, target_z),
            airsim.to_quaternion(math.radians(pitch), 0, math.radians(yaw))
        )
        print(f"[DEBUG] {name} → Teleporting to {pose_exact.position}...")
        client.simSetVehiclePose(pose_exact, ignore_collision=True, vehicle_name=name)
        time.sleep(0.1) # 短暂等待瞬移生效，这个可能需要

        # 3) (移除速度清除和 moveToPositionAsync 步骤)
        #    我们不再尝试用这些命令进行微调

        # 4) 直接命令悬停，让飞控尝试维持瞬移后的位置
        print(f"[DEBUG] {name} → Commanding hover...")
        client.hoverAsync(vehicle_name=name).join()
        # 可以增加一个等待时间让悬停稳定下来
        time.sleep(3.0)

        # 打印最终位置验证
        final_pose = client.simGetVehiclePose(vehicle_name=name)
        final_pos  = final_pose.position.to_numpy_array()
        target_pos = np.array([target_x, target_y, target_z])
        diff = final_pos - target_pos
        print(f"[DEBUG] {name} final position: {final_pos}")
        print(f"[DEBUG] {name} target position: {target_pos}")
        print(f"[DEBUG] {name} difference: {diff} (norm: {np.linalg.norm(diff):.4f})")


# ------------------ 主流程：手动采轨迹 ------------------

def record_uav0_trajectory_from_yaml(scene_path, pose_dir, tag=""):
    global offset_xy  # ★
    offset_xy = np.array([0.0, 0.0])  # 默认 0，运行时更新

    ensure_dir(pose_dir)

    # 读取 YAML 场景
    with open(scene_path, "r", encoding="utf-8") as f:
        scene = yaml.safe_load(f)

    # 初始化client
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # 初始化观察机
    setup_observer_uavs(scene,client)

    plot_uav_debug_points(client, scene)  # 观察红点

    # ---- UAV0 起飞并放到 start 位置 ----
    vehicle = "UAV0"
    center = scene["center"]
    mover  = scene["moving_drones"][0]
    start  = mover["start"]
    init_xyz = [center[i] + start[i] for i in range(3)]


    client = airsim.MultirotorClient()
    client.confirmConnection()

    # 确认连接后，检查当前已注册的Vehicles
    vehicle_names = client.listVehicles()
    print(f"[DEBUG] Detected Vehicles: {vehicle_names}")

    client.enableApiControl(True, vehicle)
    client.armDisarm(True, vehicle)
    client.takeoffAsync(vehicle_name=vehicle).join()

    client.simSetVehiclePose(
        airsim.Pose(airsim.Vector3r(*init_xyz), airsim.to_quaternion(0,0,0)),
        ignore_collision=True, vehicle_name=vehicle
    )
    client.hoverAsync(vehicle_name=vehicle).join()

    # ---- 轨迹记录准备 ----
    pose_path   = get_pose_filename(vehicle, pose_dir, tag)
    q           = Queue(maxsize=200)
    writer_proc = Process(target=pose_writer_thread, args=(q, pose_path))
    writer_proc.start()
    recording   = False

    # ---- Pygame 控制 ----
    pygame.init()
    screen = pygame.display.set_mode((420, 200))
    pygame.display.set_caption("UAV0 Manual Control")
    font  = pygame.font.Font(None, 36)
    clock = pygame.time.Clock()

    print("[INFO] V: start/stop record │ W/S: up/down │ arrows: planar │ A/D: yaw │ ESC: quit")

    try:
        while True:
            clock.tick(60)
            screen.fill((0,0,0))
            txt = font.render(f"Recording: {'YES' if recording else 'NO'}", True, (0,255,0))
            screen.blit(txt, (20,80)); pygame.display.flip()

            for event in pygame.event.get():
                if event.type in (pygame.QUIT, pygame.KEYDOWN) and (
                     event.type==pygame.QUIT or event.key==pygame.K_ESCAPE):
                    raise KeyboardInterrupt
                if event.type==pygame.KEYDOWN and event.key==pygame.K_v:
                    recording = not recording
                    print(f"[INFO] Recording {'started' if recording else 'stopped'}")

            keys = pygame.key.get_pressed()
            vx = vy = vz = yaw_rate = 0
            speed = 3; turn = 30
            if keys[pygame.K_UP]:    vx =  speed
            if keys[pygame.K_DOWN]:  vx = -speed
            if keys[pygame.K_LEFT]:  vy = -speed
            if keys[pygame.K_RIGHT]: vy =  speed
            if keys[pygame.K_w]:     vz = -speed
            if keys[pygame.K_s]:     vz =  speed
            if keys[pygame.K_a]: yaw_rate = -turn
            if keys[pygame.K_d]: yaw_rate =  turn

            # —— 保持高度：没按 W/S 就锁定当前 z ——
            if not(keys[pygame.K_w] or keys[pygame.K_s]):
                cur_z = client.simGetVehiclePose(vehicle).position.z_val
                client.moveByVelocityZAsync(
                    vx, vy, cur_z, 0.1,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                    vehicle_name=vehicle)
            else:
                client.moveByVelocityAsync(
                    vx, vy, vz, 0.1,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=yaw_rate),
                    vehicle_name=vehicle)

            # —— 记录轨迹 ——
            if recording and not q.full():
                p = client.simGetVehiclePose(vehicle)
                pos = p.position.to_numpy_array()
                ori = airsim.to_eularian_angles(p.orientation)
                ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
                q.put([ts, *pos, *ori])

    except KeyboardInterrupt:
        print("[INFO] User exit")

    finally:
        if recording:
            q.put(None); writer_proc.join()
        pygame.quit()
        client.armDisarm(False, vehicle); client.enableApiControl(False, vehicle)
        print("[INFO] UAV0 shut down")

def plot_uav_debug_points(client, scene):
    """在每架 observer 和 mover UAV 的控制点位置打上红色可视点"""
    center = scene["center"]
    all_uavs = scene.get("observer_drones", []) + scene.get("moving_drones", [])

    points = []
    for uav in all_uavs:
        if "pose" in uav:
            rel = uav["pose"]
        else:
            rel = uav["start"]
        abs_pos = [center[i] + rel[i] for i in range(3)]

        point = airsim.Vector3r(*abs_pos)
        points.append(point)
        print(f"[DEBUG] UAV {uav['name']} → Plot debug point at {abs_pos}")

    client.simPlotPoints(
        points,
        color_rgba=[1.0, 0.0, 0.0, 1.0],   # 红色点
        size=30.0,
        duration=0.0,                    # 一直显示
        is_persistent= True
    )

# ------------------ CLI ------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", required=True, help="YAML scene file")
    parser.add_argument("--tag",   default="", help="trajectory filename tag")
    args = parser.parse_args()

    ensure_dir("trajectory")
    record_uav0_trajectory_from_yaml(args.scene, "trajectory", args.tag)
