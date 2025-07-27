# main.py
import argparse
import multiprocessing as mp
import os
import time
import math

import airsim

from config import Scene
from motion import build_motion
from recorder import recorder, write_video
from uav_manager import setup_uav, set_camera
from utils import ensure_dir, log, timestamp


def run_scene(scene_path: str):
    # 加载YAML场景配置
    scene = Scene(scene_path)
    print(f"[INFO] Scene loaded: {scene.name} with {len(scene.observers)} observers, {len(scene.movers)} movers")

    # 使用 YAML 文件名作为视频文件夹名
    scene_file_base = os.path.splitext(os.path.basename(scene_path))[0]  # e.g. scene-001

    video_dir = os.path.join("./dataset", "videos", scene_file_base)
    ensure_dir(video_dir)

    print("[INFO] Connecting to AirSim...")
    client = airsim.MultirotorClient()
    client.confirmConnection()
    print("[INFO] Connected to AirSim!")

    # 尝试重置 AirSim
    try:
        print("[INFO] Attempting to reset AirSim state after connection...")
        client.reset()
        print("[INFO] AirSim reset command sent successfully.")
        # 等待一小会儿，让 AirSim 有时间完成重置
        time.sleep(2)
    except Exception as reset_err:
        # 即便重置失败也通常可以继续，但打印警告
        log(f"[WARN] Failed to reset AirSim (this might be okay): {reset_err}", "WARN")

    # 把相对位置转换为绝对地图坐标
    center = scene.center
    rel2abs = lambda v: (center[0] + v[0], center[1] + v[1], center[2] + v[2])

    # 设置观察无人机
    print("[INFO] Setting observer UAVs...")
    for obs in scene.observers:
        setup_uav(client, obs["name"], rel2abs(obs["pose"]), yaw_deg=obs.get("camera", {}).get("yaw_deg", 0))
        cam_cfg = obs.get("camera", {})
        set_camera(
            client,
            obs["name"],
            mode=cam_cfg.get("mode", "lookdown"),
            pitch_deg=cam_cfg.get("pitch_deg"),
            yaw_deg=cam_cfg.get("yaw_deg"),
        )


    # 设置运动无人机 + 启动运动线程
    print("[INFO] Setting moving UAVs...")
    motions = []
    for mover in scene.movers:
        setup_uav(client, mover["name"], rel2abs(mover["start"]))
        motions.append(build_motion(mover["name"], mover["motion"], client, center))

    print("[INFO] Starting motion threads...")
    for m in motions:
        m.start()

    # 启动多进程图像采集（每架观察无人机一个进程）
    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()
    frames_dict, fps_dict = manager.dict(), manager.dict()
    stop_flags = {obs["name"]: mp.Value("b", False) for obs in scene.observers}

    procs = [
        mp.Process(
            target=recorder,
            args=(name, scene.fps, frames_dict, fps_dict, flag),
            daemon=True,
        )
        for name, flag in stop_flags.items()
    ]

    print("[INFO] Starting recorder processes...")
    for p in procs:
        p.start()

    # 主线程等待运动结束 or Ctrl+C 中断
    log("Press Ctrl+C to stop the scene…")
    try:
        while any(m.is_alive() for m in motions):
            time.sleep(1)
    except KeyboardInterrupt:
        log("Keyboard interrupt — stopping", "WARN")

    # 停止录制进程，保存视频
    for flag in stop_flags.values():
        flag.value = True
    for p in procs:
        log(f"DEBUG: Joining recorder process {p.name}...")
        p.join()
        log(f"DEBUG: Recorder process {p.name} joined.")

    for obs in scene.observers:
        frames = frames_dict.get(obs["name"], [])
        fps_actual = fps_dict.get(obs["name"], scene.fps)
        out = os.path.join(video_dir, f"{scene_file_base}_{obs['name']}_{timestamp()}.avi")
        write_video(frames, out, fps_actual)

    # UAV 降落 & 断电
    for drone in scene.observers + scene.movers:
        name = drone["name"]
        try:
            client.hoverAsync(vehicle_name=name).join()
            client.landAsync(vehicle_name=name).join()
            client.armDisarm(False, name)
            client.enableApiControl(False, name)
        except Exception as e:
            log(f"{name}: shutdown error {e}", "WARN")

    log("Scene finished — all done!")

# def setup_external_cameras(client, external_cam_configs):
#     for cam in external_cam_configs:
#         name = cam["name"]
#         pos = cam.get("position", [0, 0, 0])
#         pitch = cam.get("pitch_deg", -20)
#         yaw = cam.get("yaw_deg", 0)
#         roll = cam.get("roll_deg", 0)
#
#         quat = airsim.to_quaternion(
#             math.radians(pitch),
#             math.radians(roll),
#             math.radians(yaw),
#         )
#         client.simSetCameraPose(name, airsim.Pose(
#             airsim.Vector3r(*pos),
#             quat
#         ))
#         print(f"[INFO] ExternalCamera {name} set at {pos}, pitch={pitch}, yaw={yaw}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AirSim multi‑UAV YAML scene")
    parser.add_argument("--scene", required=True, help="Path to YAML scene file")
    args = parser.parse_args()

    run_scene(args.scene)