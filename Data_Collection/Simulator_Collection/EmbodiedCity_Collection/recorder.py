# recorder.py
"""帧采集 & 视频写入"""
import time
from typing import Dict, List
import traceback # 用于打印更详细的错误信息

import airsim
import cv2
import numpy as np

from utils import log # 假设 log 函数来自你的 utils

__all__ = [
    "recorder",
    "write_video",
]


def recorder(name: str, fps_target: float, frames_dict: Dict, fps_dict: Dict, stop_signal):
    """
    从某架无人机的相机中采集RGB图像帧【进程级别，每个UAV单开一个】
    :param name: 无人机名字
    :param fps_target: 期望录制帧率
    :param frames_dict: 图像帧 (Manager Dict Proxy)
    :param fps_dict: 实际录制的帧率（用来写视频） (Manager Dict Proxy)
    :param stop_signal: 控制停止录制 (mp.Value)
    :return:
    """
    log(f"{name}: DEBUG - Recorder process started. Target FPS: {fps_target:.2f}")
    client = airsim.MultirotorClient()
    try:
        client.confirmConnection()
        log(f"{name}: DEBUG - Connected to AirSim.")
    except Exception as conn_err:
        log(f"{name}: ERROR - Failed to connect to AirSim: {conn_err}", "ERROR")
        log(f"{name}: DEBUG - Recorder function exiting due to connection error.")
        return # 连接失败则直接退出进程

    frames: List[np.ndarray] = []   # 存储图像帧
    frame_count = 0
    t0 = time.time()    # 起始时间
    log(f"{name}: DEBUG - Entering recording loop.")

    loop_iteration = 0 # 循环计数器
    try: # 包裹主循环和后续处理，以便捕获意外错误
        while not stop_signal.value:
            loop_iteration += 1
            # --- 循环开始日志 ---
            log(f"{name}: DEBUG - Loop iteration {loop_iteration}. Stop signal: {stop_signal.value}")

            try:
                # --- 获取图像前的日志 ---
                log(f"{name}: DEBUG - Calling simGetImages...")
                response_list = client.simGetImages([
                    airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)  # 注意不压缩
                ], vehicle_name=name)

                if not response_list:
                     log(f"{name}: WARN - simGetImages returned an empty list.", "WARN")
                     response = None
                else:
                     response = response_list[0]

                # --- 获取图像后的日志 ---
                if response:
                    log(f"{name}: DEBUG - simGetImages returned. Response width: {response.width}, Height: {response.height}")
                    # 解码图像并保存 (增加更严格的检查)
                    if response.width > 0 and response.height > 0 and len(response.image_data_uint8) > 0 :
                        try:
                            frame = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(
                                response.height, response.width, 3
                            )
                            frames.append(frame)
                            frame_count += 1
                            log(f"{name}: DEBUG - Frame {frame_count} added.")
                        except ValueError as reshape_err:
                             log(f"{name}: ERROR - Failed to reshape image data. Size: {len(response.image_data_uint8)}, Expected: {response.height}*{response.width}*3. Error: {reshape_err}", "ERROR")
                        except Exception as decode_err:
                             log(f"{name}: ERROR - Failed to decode/append frame: {decode_err}", "ERROR")
                    else:
                        log(f"{name}: WARN - Received empty/invalid image data (Width={response.width}, Height={response.height}, DataSize={len(response.image_data_uint8)}).", "WARN")
                else:
                    log(f"{name}: WARN - No valid response from simGetImages this iteration.", "WARN")

            except Exception as img_err:
                # 捕获 simGetImages 或图像处理中的错误
                log(f"{name}: ERROR - Error during simGetImages or image processing: {img_err}", "ERROR")
                log(f"{name}: ERROR - Traceback: {traceback.format_exc()}")
                # 可以选择是退出循环还是等待后重试
                # break # 如果希望出错时停止录制，取消这行注释
                time.sleep(0.5) # 等待一小会再试

            # 采样帧率控制
            sleep_duration = (1.0 / fps_target) * 0.7
            log(f"{name}: DEBUG - Sleeping for {sleep_duration:.3f} seconds.")
            time.sleep(sleep_duration)

        # --- 循环结束 ---
        log(f"{name}: DEBUG - Recording loop finished. Stop signal received (value: {stop_signal.value}).")

        elapsed = time.time() - t0
        actual_fps = frame_count / elapsed if elapsed > 0 else fps_target
        log(f"{name}: DEBUG - Calculated actual FPS: {actual_fps:.2f} ({frame_count} frames in {elapsed:.2f}s)")

        # --- 写回 Manager 字典前的日志 ---
        log(f"{name}: DEBUG - Preparing to write results to Manager dicts. Frame count: {frame_count}")
        try:
            # --- 写 frames_dict ---
            log(f"{name}: DEBUG - Writing {frame_count} frames to frames_dict...")
            frames_dict[name] = frames # <--- 潜在卡点 1
            log(f"{name}: DEBUG - Frames written to frames_dict successfully.")

            # --- 写 fps_dict ---
            log(f"{name}: DEBUG - Writing actual FPS ({actual_fps:.2f}) to fps_dict...")
            fps_dict[name] = actual_fps # <--- 潜在卡点 2 (可能性小)
            log(f"{name}: DEBUG - FPS written to fps_dict successfully.")

        except Exception as dict_err:
            log(f"{name}: ERROR - Error writing results to Manager dict: {dict_err}", "ERROR")
            log(f"{name}: ERROR - Traceback: {traceback.format_exc()}")


        log(f"{name}: INFO - Recording stopped summary: frames={frame_count} · fps={actual_fps:.2f}") # 保留原 INFO 日志

    except Exception as main_err:
        # 捕获主逻辑中的意外错误
        log(f"{name}: ERROR - An unexpected error occurred in the recorder main logic: {main_err}", "ERROR")
        log(f"{name}: ERROR - Traceback: {traceback.format_exc()}")
    finally:
        # --- 确保函数退出前打印日志 ---
        log(f"{name}: DEBUG - Recorder function is exiting.")


# --- write_video 函数保持不变 ---
def write_video(frames: List[np.ndarray], out_path: str, fps: float):
    """
    将采集到的图像帧写成一个.avi视频文件
    :param frames: 图像帧列表
    :param out_path: 视频保存路径
    :param fps: 视频帧率
    :return:
    """
    if not frames:
        log(f"WARN - No frames captured for {out_path}. Skipping video writing.", "WARN")
        return
    try:
        height, width, _ = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        log(f"INFO - Writing video {out_path} ({len(frames)} frames, {width}x{height} @ {fps:.2f} FPS)")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        for i, f in enumerate(frames):
            if f.shape[0] != height or f.shape[1] != width:
                log(f"WARN - Resizing frame {i} from {f.shape[1]}x{f.shape[0]} to {width}x{height} for {out_path}", "WARN")
                f = cv2.resize(f, (width, height))
            writer.write(f)
        writer.release()
        log(f"INFO - Video saved successfully → {out_path}")
    except Exception as write_err:
        log(f"ERROR - Error writing video file {out_path}: {write_err}", "ERROR")
        log(f"ERROR - Traceback: {traceback.format_exc()}")