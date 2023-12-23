from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch
from smartargparse import BaseConfig, parse_args
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from modules.topological_map import TopologicalMap
import random
import torch.nn.functional as F


@dataclass(frozen=True)
class Config(BaseConfig):
    topological_map_path: str
    compare_map_path: str = ""
    seed: int = 42

class FeatureVisualizer:
    _config: Config
    _topological_map: TopologicalMap

    def __init__(self) -> None:
        self._config = parse_args(Config)
        self._topological_map = TopologicalMap.load(self._config.topological_map_path)
        self._compare_map = None if self._config.compare_map_path=="" else TopologicalMap.load(self._config.compare_map_path)

    def __call__(self) -> None:
        fig = plt.figure()
        vector3d, color_map = self.get_embedded_vector(self._topological_map.summalize_features, dim=3, print_option=True)
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        self.plot_vector3d(ax, vector3d, color_map, cm.hsv)

        if self._compare_map is not None:
            vec_compare, cmap_compare = self.get_embedded_vector(self._compare_map.summalize_features, dim=3, print_option=True)
            self.plot_vector3d(ax, vec_compare, cmap_compare, cm.twilight)

        plt.show()

    def get_embedded_vector(self, map_features, dim, print_option):
        tsne = TSNE(n_components=dim, random_state=self._config.seed)
        features = []
        cmap = []
        if print_option:
            print("==================================")
        for i, feature in enumerate(map_features):
            if print_option:
                print(f"{i}: {F.cosine_similarity(map_features[0], feature, dim=0).item()}")
            vec = feature.to('cpu').detach().numpy().copy()
            cmap.append(i/len(map_features))
            features.append(vec)
        np_features = np.array(features)
        embedded_vector = tsne.fit_transform(np_features)

        return embedded_vector, cmap

    def plot_vector3d(self, ax, vec, cmap, color) -> None:
        # for i, (x,y,z) in enumerate(vec):
        #     ax.text(x, y, z, f"{i}", color="gray")
        ax.scatter(vec[:, 0], vec[:, 1], vec[:, 2], c=color(cmap))

    def plot_vector2d(self, figure, vec, cmap) -> None:
        plt.xlabel('X')
        plt.ylabel('Y')
        for i, (x,y) in enumerate(vector2d):
            plt.text(x, y, f"{i}")
        plt.scatter(vector2d[:, 0], vector2d[:, 1], c=cm.hsv(cmap))
