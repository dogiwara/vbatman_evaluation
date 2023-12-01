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
    seed: int = 42

class FeatureVisualizer:
    _config: Config
    _topological_map: TopologicalMap

    def __init__(self) -> None:
        self._config = parse_args(Config)
        self._topological_map = TopologicalMap.load(self._config.topological_map_path)

    def __call__(self) -> None:
        tsne_3d = TSNE(n_components=3, random_state=self._config.seed)
        tsne_2d = TSNE(n_components=2, random_state=self._config.seed)
        features = []
        color_map = []
        print(len(self._topological_map.summalize_features[0]))

        for i, feature in enumerate(self._topological_map.summalize_features):
            print(f"{i}: {F.cosine_similarity(self._topological_map.summalize_features[0], feature, dim=0).item()}")
            vec = feature.to('cpu').detach().numpy().copy()
            color_map.append(i/len(self._topological_map.summalize_features))
            features.append(vec)
        features = np.array(features)
        embedded_vector_2d = tsne_2d.fit_transform(features)
        embedded_vector_3d = tsne_3d.fit_transform(features)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        for i, (x,y,z) in enumerate(embedded_vector_3d):
            ax.text(x, y, z, f"{i}", color="gray")
        ax.scatter(embedded_vector_3d[:, 0], embedded_vector_3d[:, 1], embedded_vector_3d[:, 2], c=cm.hsv(color_map))
        plt.show()

        plt.xlabel('X')
        plt.ylabel('Y')
        for i, (x,y) in enumerate(embedded_vector_2d):
            plt.text(x, y, f"{i}")
        plt.scatter(embedded_vector_2d[:, 0], embedded_vector_2d[:, 1], c=cm.hsv(color_map))
        plt.show()

