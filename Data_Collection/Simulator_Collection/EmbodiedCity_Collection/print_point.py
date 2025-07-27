import airsim
import pygame
import time
# import csv # 不再需要 CSV
import os
from datetime import datetime

# def ensure_dir(path): # 如果不保存任何文件，可能不再需要
#     if not os.path.exists(path):
#         os.makedirs(path)

# def get_timestamp(): # 如果不用于文件名，可能不再需要
#     return datetime.now().strftime('%Y%m%d_%H%M%S')

# def save_pose(vehicle_name, client, writer): # 不再需要记录轨迹
#     pose = client.simGetVehiclePose(vehicle_name=vehicle_name)
#     position = pose.position.to_numpy_array()
#     orientation = airsim.to_eularian_angles(pose.orientation)
#     timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
#     writer.writerow([timestamp, *position, *orientation])
#     print(f"[INFO] {vehicle_name} 位置记录: {position}")

def get_and_print_uav0_pose(client):
    """获取UAV0的位姿并打印坐标"""
    try:
        pose = client.simGetVehiclePose(vehicle_name="UAV0")
        position = pose.position.to_numpy_array()
        # orientation = airsim.to_eularian_angles(pose.orientation) # 如果只需要坐标，可以注释掉姿态
        print(f"[INFO] UAV0 当前坐标 (X, Y, Z): {position}")
    except Exception as e:
        print(f"[ERROR] 获取 UAV0 位姿失败: {e}")

def manual_control():
    # 初始化 AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()

    vehicle_names = ["UAV0", "UAV1", "UAV2", "UAV3"]
    # 注意：这里的坐标是硬编码的，如果需要从YAML加载，需要结合之前的config.py等
    positions = [
        (6211.73, -4243.8447, -3.8462045), # UAV0
        (6215.00, -4240.00, -30.0),        # UAV1
        (6220.00, -4250.00, -30.0),        # UAV2
        (6200.00, -4255.00, -30.0)         # UAV3
    ]
    yaw = 1.55334302  # 大约90度 (弧度)

    # 初始化无人机位置（与原代码相同）
    for idx, vehicle_name in enumerate(vehicle_names):
        try:
            client.enableApiControl(True, vehicle_name=vehicle_name)
            client.armDisarm(True, vehicle_name=vehicle_name)
            print(f"[INFO] Taking off {vehicle_name}...")
            client.takeoffAsync(vehicle_name=vehicle_name).join()
            print(f"[INFO] {vehicle_name} Takeoff complete.")

            pose = airsim.Pose(
                airsim.Vector3r(*positions[idx]),
                airsim.to_quaternion(0, 0, yaw) # 所有无人机初始偏航角相同
            )
            print(f"[INFO] Setting pose for {vehicle_name} to {positions[idx]}...")
            client.simSetVehiclePose(pose, ignore_collision=True, vehicle_name=vehicle_name)
            time.sleep(0.1) # 瞬移后短暂等待

            print(f"[INFO] Hovering {vehicle_name}...")
            client.hoverAsync(vehicle_name=vehicle_name).join()
            time.sleep(1.0) # 等待悬停稳定
            print(f"[INFO] {vehicle_name} setup complete.")
        except Exception as e:
            print(f"[ERROR] Setting up {vehicle_name} failed: {e}")
            # 可以选择退出或继续设置其他无人机
            # return

    # 初始化 Pygame
    pygame.init()
    if not pygame.get_init():
        print("[ERROR] Pygame 初始化失败！")
        return # Pygame 失败则退出
    else:
        print("[INFO] Pygame 初始化成功，窗口创建中...")

    try:
        screen = pygame.display.set_mode((450, 150)) # 稍微加宽窗口以显示新文本
        pygame.display.set_caption('UAV0 手动控制')
        font = pygame.font.Font(None, 32)
        clock = pygame.time.Clock()
    except pygame.error as e:
        print(f"[ERROR] 创建 Pygame 窗口失败: {e}")
        pygame.quit()
        return

    speed = 3.0
    turn_rate = 30
    control_command = {'vx': 0.0, 'vy': 0.0, 'vz': 0.0, 'yaw_rate': 0.0}

    # --- 移除轨迹记录相关的变量 ---
    # recording = False
    # csv_writer = None
    # csv_file = None
    # last_record_time = time.time()
    # trajectory_dir = "dataset/trajectory"
    # ensure_dir(trajectory_dir)
    # --- 移除结束 ---

    try:
        running = True
        while running:
            clock.tick(60) # 限制帧率

            # --- 更新屏幕显示文本 ---
            screen.fill((0, 0, 0))
            status_text = f'UAV0 手动控制中'
            text = font.render(status_text, True, (255, 255, 255))
            screen.blit(text, (20, 50))
            instruction = font.render('按 F 键 打印UAV0坐标', True, (255, 255, 255)) # 修改提示
            screen.blit(instruction, (20, 90))
            pygame.display.flip()
            # --- 更新结束 ---

            # 处理事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    print("[INFO] Pygame 窗口关闭，准备退出...")

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: # 添加 ESC 键退出
                        running = False
                        print("[INFO] ESC 键按下，准备退出...")
                    elif event.key == pygame.K_f:
                        # --- 修改 F 键功能 ---
                        # 不再切换记录状态，而是直接获取并打印坐标
                        print("[INFO] F 键按下，获取 UAV0 坐标...")
                        get_and_print_uav0_pose(client)
                        # --- 修改结束 ---

            # 如果窗口已关闭，则停止循环 (防止 get_pressed 报错)
            if not running:
                break

            # 获取按键状态并更新控制指令
            try:
                keys = pygame.key.get_pressed()
                # --- 控制逻辑（与原代码相同）---
                control_command['vx'] = speed if keys[pygame.K_UP] else (-speed if keys[pygame.K_DOWN] else 0)
                control_command['vy'] = speed if keys[pygame.K_RIGHT] else (-speed if keys[pygame.K_LEFT] else 0)
                control_command['vz'] = -speed if keys[pygame.K_w] else (speed if keys[pygame.K_s] else 0) # W上升, S下降 (因为Z轴向下为正)
                control_command['yaw_rate'] = turn_rate if keys[pygame.K_d] else (-turn_rate if keys[pygame.K_a] else 0)

                # 发送控制指令或悬停
                if (control_command['vx'] == 0 and control_command['vy'] == 0 and
                    control_command['vz'] == 0 and control_command['yaw_rate'] == 0):
                    # 如果没有按键，发送悬停指令（可以考虑不频繁发送，比如仅在状态改变时发送一次）
                    # 为了简单，这里保持每次循环检查
                    client.hoverAsync(vehicle_name="UAV0") # hoverAsync 不需要 join
                else:
                    # 发送速度控制指令
                    client.moveByVelocityAsync(
                        control_command['vx'],
                        control_command['vy'],
                        control_command['vz'],
                        duration=0.1, # 持续时间短，依赖循环刷新
                        drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                        yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=control_command['yaw_rate']),
                        vehicle_name="UAV0"
                    )
                # --- 控制逻辑结束 ---
            except pygame.error as e:
                 # 如果在获取按键时 Pygame 窗口已关闭，可能会出错
                 print(f"[WARN] Pygame 错误 (可能窗口已关闭): {e}")
                 running = False # 标记退出
            except Exception as e:
                 print(f"[ERROR] 控制无人机时出错: {e}")
                 # 可以选择继续运行或标记退出
                 # running = False

            # --- 移除按时间间隔记录轨迹的部分 ---
            # if recording and csv_writer and time.time() - last_record_time > 0.5:
            #     save_pose("UAV0", client, csv_writer)
            #     last_record_time = time.time()
            # --- 移除结束 ---

            # 短暂休眠，避免 CPU 占用过高
            time.sleep(0.01)


    # --- 清理部分 ---
    finally:
        # --- 移除关闭 CSV 文件的部分 ---
        # if csv_file:
        #     csv_file.close()
        # --- 移除结束 ---

        # 退出 Pygame
        if pygame.get_init():
            pygame.quit()
            print("[INFO] Pygame 已退出。")

        # 锁定无人机并释放控制权
        print("[INFO] 正在锁定并释放无人机控制权...")
        for vehicle_name in vehicle_names:
            try:
                # 尝试先悬停再降落可能更安全，但如果只是退出控制，直接锁定即可
                client.armDisarm(False, vehicle_name=vehicle_name)
                client.enableApiControl(False, vehicle_name=vehicle_name)
                print(f"[INFO] {vehicle_name} 已锁定并释放控制。")
            except Exception as e:
                print(f"[WARN] 清理 {vehicle_name} 时出错: {e}")

        print("[INFO] UAV 手动控制程序已安全退出。")

if __name__ == "__main__":
    manual_control()