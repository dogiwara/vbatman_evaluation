import pathlib
import random
import sys
from dataclasses import dataclass
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
    transform: str
    n_trial: int
    img_dir: str = "/tmp/map_evaluator"
    seed: int = 42
    show_img: bool = False
    save_img: bool = False


@dataclass(frozen=True)
class GridMap:
    data: np.ndarray
    resolution: float
    origin: Tuple[float, float, float]


class TopologicalMapEvaluator:
    _config: Config
    _eval_map: TopologicalMap
    _grid_map: GridMap
    _n_success: int

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
        self._n_success = 0

        random.seed(self._config.seed)
        if self._config.save_img:
            pathlib.Path(self._config.img_dir).mkdir(parents=True, exist_ok=True)

    def __call__(self) -> None:
        for i in range(self._config.n_trial):
            node_len = len(self._eval_map.nodes)
            source, target = 0, 0
            while source == target:
                source = random.randint(0, node_len - 1)
                target = random.randint(0, node_len - 1)

            eval_path = cast(
                List[int],
                nx.shortest_path(
                    self._eval_map.graph, source=source, target=target, weight="weight", method="dijkstra"))
            eval_poses = [
                self._eval_map.gt_poses[node_id][:2] for node_id in eval_path]
            is_success = self.is_path_valid(eval_poses)
            if is_success:
                self._n_success += 1

            if not self._config.show_img and not self._config.save_img:
                continue
            path_image = eval(f"self.transform_{self._config.transform}")(eval_poses)  # Call transform method from given name
            if self._config.save_img:
                cv2.imwrite(f"{self._config.img_dir}/{i}_{is_success}.png", path_image)
            if self._config.show_img:
                cv2.imshow("path on grid map", path_image)
                while cv2.waitKey(1) == -1:
                    pass

        print(f"Success rate: {self._n_success / self._config.n_trial}")

    def coordinate_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        map_height = self._grid_map.data.shape[0]

        return (int((x - self._grid_map.origin[0]) / self._grid_map.resolution),
                int(map_height - (y - self._grid_map.origin[1]) / self._grid_map.resolution))

    def is_path_valid(self, poses: List[Tuple[float, float]]) -> bool:
        for i in range(len(poses) - 1):
            start = self.coordinate_to_pixel(*poses[i])
            end = self.coordinate_to_pixel(*poses[i + 1])
            if not self.is_line_valid(start, end):
                return False
        return True

    def is_line_valid(self, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        x1, y1 = start
        x2, y2 = end
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy

        while True:
            if self._grid_map.data[y1, x1] != 254:
                return False
            if x1 == x2 and y1 == y2:
                break
            e2 = err * 2
            if e2 > -dy:
                err -= dy
                x1 += sx
            if e2 < dx:
                err += dx
                y1 += sy
        return True

    def draw_path_on_grid_map(self, path: List[Tuple[float, float]]) -> np.ndarray:
        map_img = cv2.cvtColor(self._grid_map.data, cv2.COLOR_GRAY2BGR)
        for i in range(len(path)):
            point = self.coordinate_to_pixel(*path[i])
            if i < len(path) - 1:
                start = point
                end = self.coordinate_to_pixel(*path[i + 1])
                cv2.line(map_img, start, end, color=(255, 0, 0), thickness=10)
            cv2.circle(map_img, point, radius=8, color=(0, 255, 0), thickness=-1)

        return map_img

    def transform_dkan(self, eval_poses: List[Tuple[float, float]]) -> np.ndarray:
        path_image = self.draw_path_on_grid_map(eval_poses)[550 * 3:780 * 3, 430 * 3:900 * 3, :]

        return path_image

    def transform_dkan2f(self, eval_poses: List[Tuple[float, float]]) -> np.ndarray:
        path_image = self.draw_path_on_grid_map(eval_poses)
        trans = cv2.getRotationMatrix2D((path_image.shape[1] // 2, path_image.shape[0] // 2), -1.5, 1)
        path_image = cv2.warpAffine(
            path_image,
            trans,
            (path_image.shape[1], path_image.shape[0]))[
            600 * 3:830 * 3, 430 * 3:900 * 3, :]

        return path_image

    def transform_dkan_around(self, eval_poses: List[Tuple[float, float]]) -> np.ndarray:
        path_image = self.draw_path_on_grid_map(eval_poses)[1650:3150, 3500:5300, :]
        # trans = cv2.getRotationMatrix2D((path_image.shape[1] // 2, path_image.shape[0] // 2), 0, 1)
        # path_image = cv2.warpAffine(
        #     path_image,
        #     trans,
        #     (path_image.shape[1], path_image.shape[0]))[1650:3150, 3500:5300, :]
        path_image = cv2.resize(path_image, None, None, 0.4, 0.4)

        return path_image
