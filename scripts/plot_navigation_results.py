#!/usr/bin/env python3

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from matplotlib import pyplot as plt
from smartargparse import BaseConfig, parse_args


@dataclass(frozen=True)
class Config(BaseConfig):
    vbatman_log_path: str
    ving_log_path: str
    save_dir: str = "/tmp/navigation_results"
    dist_range: float = 5


def calc_success_rates(
        config: Config, log_path: str) -> Tuple[List[float], List[float], float]:
    success_rates = [1.0]
    ranges = [0.0]
    n_all_sccuess = 0
    n_all_data = 0
    with Path(log_path).open("r") as f:
        reader = csv.reader(f, delimiter=",")
        dists = []
        successes = []
        for dist, success in reader:
            dists.append(float(dist))
            successes.append(success == " True")

        n_all_data = len(dists)
        n_itr = math.ceil(max(dists) / config.dist_range)
        for i in range(n_itr):
            range_min = i * config.dist_range
            range_max = (i + 1) * config.dist_range
            n_data = 0
            n_success = 0
            for j, dist in enumerate(dists):
                if range_min <= dist < range_max:
                    n_data += 1
                    if successes[j]:
                        n_success += 1
                        n_all_sccuess += 1
            if n_data == 0:
                continue
            ranges.append(range_max)
            success_rate = n_success / n_data
            success_rates.append(success_rate)

    return ranges, success_rates, n_all_sccuess / n_all_data


def main() -> None:
    config = parse_args(Config)
    Path(config.save_dir).mkdir(parents=True, exist_ok=True)

    plt.rcParams["font.family"] = "Roboto"
    plt.rcParams["font.size"] = 15
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["legend.fancybox"] = False
    plt.rcParams["legend.framealpha"] = 1.0
    plt.rcParams["legend.edgecolor"] = "black"
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, xlabel="Goal Distance [m]", ylabel="Success Rate")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 1.0)

    vbatman_ranges, vbatman_success_rates, vbatman_all_success_rate = calc_success_rates(config, config.vbatman_log_path)
    ving_ranges, ving_success_rates, ving_all_success_rate = calc_success_rates(config, config.ving_log_path)
    print(f"VBATMAN: {vbatman_all_success_rate}, ViNG: {ving_all_success_rate}")

    ax.plot(vbatman_ranges, vbatman_success_rates, label="VBATMAN", linestyle="-")
    ax.plot(ving_ranges, ving_success_rates, label="ViNG-like", linestyle="-.")
    ax.legend(loc="upper right")
    plt.savefig(Path(config.save_dir) / "navigation_results.svg",
                bbox_inches="tight", pad_inches=0.05)
    plt.show()


if __name__ == "__main__":
    main()
