import pathlib
import sys
from dataclasses import dataclass
from random import randint
from typing import List, Tuple, cast

import cv2
import networkx as nx
import numpy as np
import yaml
from smartargparse import BaseConfig, parse_args

sys.path.append("/home/amsl/catkin_ws/src/vbatman/vbatman/scripts")
from modules.topological_map import TopologicalMap


@dataclass(frozen=True)
class Config(BaseConfig):
    evaluating_map_path: str
    grid_map_info_path: str
    output_path: str
    n_trial: int


@dataclass(frozen=True)
class GridMap:
    data: np.ndarray
    resolution: float
    origin: Tuple[float, float, float]


class TopologicalMapEvaluator:
    _config: Config
    _eval_map: TopologicalMap
    _grid_map: GridMap

    def __init__(self) -> None:
        self._config = parse_args(Config)
        self._eval_map = TopologicalMap.load(self._config.evaluating_map_path)
        with open(self._config.grid_map_info_path, "r") as f:
            grid_map_info = yaml.safe_load(f)
        self._grid_map = GridMap(
            data=cv2.imread(
                (pathlib.Path(self._config.grid_map_info_path).parent / grid_map_info["image"]).as_posix(),
                cv2.IMREAD_GRAYSCALE),
            resolution=grid_map_info["resolution"],
            origin=grid_map_info["origin"],
        )

    def __call__(self) -> None:
        for _ in range(self._config.n_trial):
            node_len = len(self._eval_map.nodes)
            source, target = 0, 0
            while source == target:
                source = randint(0, node_len - 1)
                target = randint(0, node_len - 1)

            eval_path = cast(
                List[int],
                nx.shortest_path(
                    self._eval_map.graph, source=source, target=target, weight="weight", method="dijkstra"))
            eval_poses = [
                self._eval_map.gt_poses[node_id][:2] for node_id in eval_path]
            path_image = self.draw_path_on_grid_map(eval_poses)
            cv2.imshow(
                "path on grid map",
                cv2.resize(path_image, (path_image.shape[1] // 3, path_image.shape[0] // 3)))
            cv2.waitKey(0)

    def coordinate_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        return (int((x - self._grid_map.origin[0]) / self._grid_map.resolution),
                int((y - self._grid_map.origin[1]) / self._grid_map.resolution))

    def draw_path_on_grid_map(self, path: List[Tuple[float, float]]) -> np.ndarray:
        map_img = cv2.cvtColor(self._grid_map.data, cv2.COLOR_GRAY2BGR)
        for i in range(len(path)):
            point = self.coordinate_to_pixel(*path[i])
            cv2.circle(map_img, point, radius=10, color=(0, 255, 0), thickness=-1)
            if i < len(path) - 1:
                start = point
                end = self.coordinate_to_pixel(*path[i + 1])
                cv2.line(map_img, start, end, color=(255, 0, 0), thickness=10)

        return map_img
