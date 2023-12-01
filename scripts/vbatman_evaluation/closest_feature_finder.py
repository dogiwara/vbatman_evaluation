from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from smartargparse import BaseConfig, parse_args
from modules.topological_map import TopologicalMap
import torch.nn.functional as F
from typing import List

@dataclass(frozen=True)
class Config(BaseConfig):
    topological_map_path: str
    positive_nodes: List[int]
    negative_nodes: List[int]
    seed: int = 42

class ClosestFeatureFinder:
    _config: Config
    _topological_map: TopologicalMap

    def __init__(self) -> None:
        self._config = parse_args(Config)
        self._topological_map = TopologicalMap.load(self._config.topological_map_path)

    def __call__(self) -> None:
        print(f"Map has {len(self._topological_map.features)} features.")
        self.find_closest_node()

    def find_closest_node(self) -> None:
        node_features = self._topological_map.summalize_features
        calculated_feature = torch.zeros(len(node_features[0]), device=node_features[0].device)
        similarity = {}
        for i in self._config.positive_nodes:
            calculated_feature = torch.add(calculated_feature, node_features[i])
        for i in self._config.negative_nodes:
            calculated_feature = torch.add(calculated_feature, -node_features[i])

        for i, feature in enumerate(node_features):
            similarity.setdefault('{:d}'.format(i) ,F.cosine_similarity(calculated_feature, feature, dim=0).item())

        similar_ids = sorted(similarity.items(), key = lambda sim : sim[1], reverse=True)
        for i in range(10):
            print(similar_ids[i])
