"""
解析YAML场景文件，生成Scene对象
"""

import os
import yaml
from typing import Dict, Any, List, Tuple

class Scene:
    """解析YAML场景，包装成对象"""

    def __init__(self, path: str):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as f:
            raw: Dict[str, Any] = yaml.safe_load(f)

        self.name: str = raw.get("scene_name", os.path.splitext(os.path.basename(path))[0])
        self.fps: float = raw.get("record_fps", 30)
        self.center: Tuple[float, float, float] = tuple(raw.get("center", [0, 0, 0]))  # x, y, z
        self.observers: List[Dict[str, Any]] = raw.get("observer_drones", [])
        self.movers: List[Dict[str, Any]] = raw.get("moving_drones", [])
        # self.external_cameras = raw.get("external_cameras", [])

    def __repr__(self):
        return f"<Scene {self.name} | {len(self.observers)} observers, {len(self.movers)} movers>"
