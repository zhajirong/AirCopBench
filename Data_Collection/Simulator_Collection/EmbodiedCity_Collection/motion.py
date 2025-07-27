"""调整无人机的运动"""
import math
import threading
import time
from typing import Dict, Any

import airsim
import numpy as np
import pandas as pd

from utils import log

__all__ = [
    "BaseMotion",
    "CircleMotion",
    "CSVPathMotion",
    "build_motion",
]


class BaseMotion(threading.Thread):
    """所有轨迹都继承自BaseMotion，真正的轨迹逻辑放在子类的run()"""

    def __init__(self, name: str, client: airsim.MultirotorClient):
        super().__init__(daemon=True)
        self.name = name
        self.client = client    # 共用的airsim.MultirotorClient
    # 子类需要实现run()


class CircleMotion(BaseMotion):
    """圆周运动"""

    def __init__(self, name: str, client: airsim.MultirotorClient, center, radius: float, height: float,
                 angular_speed_dps: float,  rounds=1 ):
        super().__init__(name, client)
        self.center = np.array(center, dtype=float) # 圆心坐标
        self.radius = radius  # 圆半径
        self.height = height  # 在AirSim里，z轴负值向上
        self.omega = math.radians(angular_speed_dps)  # 角速度 rad/s
        self.rounds = rounds

    def run(self):
        log(f"{self.name}: circle motion started for {self.rounds} round(s)")
        t0 = time.time()
        update = 0.05

        duration = self.rounds * (2 * math.pi) / self.omega
        end_time = t0 + duration

        while time.time() < end_time:
            ang = self.omega * (time.time() - t0)
            vx = -self.radius * self.omega * math.sin(ang)
            vy = self.radius * self.omega * math.cos(ang)
            yaw = math.degrees(ang + math.pi / 2)

            try:
                self.client.moveByVelocityAsync(
                    vx, vy, 0.0, update,
                    drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                    yaw_mode=airsim.YawMode(False, yaw),
                    vehicle_name=self.name,
                )
            except Exception as e:
                log(f"[ERROR] {self.name}: {e}", level="ERROR")
                break

            time.sleep(update)

        self.client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=self.name).join()
        self.client.hoverAsync(vehicle_name=self.name).join()
        log(f"{self.name}: completed {self.rounds} full circle(s) and stopped")


class CSVPathMotion(BaseMotion):
    """根据csv文件轨迹复现无人机运动（csv中包含 timestamp, x, y, z, pitch, roll, yaw）"""

    def __init__(self, name: str, client: airsim.MultirotorClient, csv_path: str):
        super().__init__(name, client)
        df = pd.read_csv(csv_path)
        if not {"timestamp", "x", "y", "z"}.issubset(df.columns):
            raise ValueError("CSV must include timestamp,x,y,z columns")
        self.df = df
        self.times = pd.to_datetime(df["timestamp"])
        self.start_time = None

    def run(self):
        log(f"{self.name}: CSV path motion started")
        df = self.df
        total = len(df)

        for i in range(total - 1):
            cur = df.iloc[i]
            nxt = df.iloc[i + 1]
            t_cur = self.times[i]
            t_nxt = self.times[i + 1]

            dt = (t_nxt - t_cur).total_seconds()
            if dt <= 0:
                continue

            # 计算速度向量
            vx = (nxt["x"] - cur["x"]) / dt
            vy = (nxt["y"] - cur["y"]) / dt
            vz = (nxt["z"] - cur["z"]) / dt
            yaw = math.degrees(nxt.get("yaw", 0))

            self.client.moveByVelocityAsync(
                vx, vy, vz, dt,
                drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                yaw_mode=airsim.YawMode(False, yaw),
                vehicle_name=self.name
            )
            # 睡眠 ≈ dt（允许连续发送指令）
            time.sleep(dt)

        # 停止 UAV
        self.client.moveByVelocityAsync(0, 0, 0, 1, vehicle_name=self.name).join()
        self.client.hoverAsync(vehicle_name=self.name).join()
        log(f"{self.name}: CSV path motion finished")


# ---------------------------------------------------------------------------
# 工厂模式
# ---------------------------------------------------------------------------

# 映射表（运动类型）
_MOTION_REGISTRY = {
    "circle": CircleMotion,
    "csv": CSVPathMotion,
}

def build_motion(name: str, cfg: Dict[str, Any], client: airsim.MultirotorClient, center):
    # cfg 是YAML配置文件中对应该无人机的motion:字典
    mtype = cfg.get("type")
    if mtype not in _MOTION_REGISTRY:
        raise ValueError(f"Unsupported motion type: {mtype}")
    if mtype == "circle":
        return CircleMotion(
            name, client, center,
            cfg["radius"], cfg["height"], cfg["angular_speed_dps"],
            rounds=cfg.get("rounds", 1)  # 读取 rounds，默认为 1 圈
        )
    if mtype == "csv":
        return CSVPathMotion(name, client, cfg["file"])
    raise RuntimeError
