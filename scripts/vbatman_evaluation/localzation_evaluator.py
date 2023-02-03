import math
from pathlib import Path
from typing import Dict, List, Tuple, cast

import message_filters
import numpy as np
import rospy
from geometry_msgs.msg import (Pose, PoseArray, PoseStamped,
                               PoseWithCovarianceStamped)
from matplotlib import pyplot as plt
from nav_msgs.msg import Odometry
from std_msgs.msg import Int32
from transformutils import (calc_relative_pose, calc_transformed_pose,
                            get_array_2d_from_msg, get_msg_from_array_2d)
from vbatman.msg import Node, NodeArray


class LocalizationEvaluator:
    _save_dir: str
    _n_observations: int
    _init_pose: Tuple[float, float, float]
    _init_odom: Pose
    _count_steps: int
    _gt_poses: Dict[str, List[float]]
    _localized_poses: Dict[str, List[float]]
    _comparison_poses: Dict[str, List[float]]
    _odom_poses: Dict[str, List[float]]
    _sum_position_proposed: float
    _sum_orientation_proposed: float
    _sum_position_comparison: float
    _sum_orientation_comparison: float
    _proposed_errors_position: List[float]
    _proposed_errors_orientation: List[float]
    _comparison_errors_position: List[float]
    _comparison_errors_orientation: List[float]
    _evaluated: bool
    _gt_sub: message_filters.Subscriber
    _localized_pose_sub: message_filters.Subscriber
    _reference_nodes_sub: message_filters.Subscriber
    _current_index_sub: message_filters.Subscriber
    _odom_sub: message_filters.Subscriber
    _synchronizer: message_filters.ApproximateTimeSynchronizer

    def __init__(self) -> None:
        rospy.init_node("localization_evaluator")

        self._save_dir = cast(str, rospy.get_param("~save_dir"))
        Path(self._save_dir).mkdir(parents=True, exist_ok=True)
        self._n_observations = rospy.get_param(
            "/vbatman/path_handler/reference_nodes_range") * 2 + 1  # type: ignore
        self._init_pose = (  # type: ignore
            rospy.get_param("/vbatman/localizer/init_x"),
            rospy.get_param("/vbatman/localizer/init_y"),
            rospy.get_param("/vbatman/localizer/init_yaw"))
        self._init_odom = Pose()

        self._count_steps = 0
        self._gt_poses = {"x": [], "y": [], "yaw": []}
        self._localized_poses = {"x": [], "y": [], "yaw": []}
        self._comparison_poses = {"x": [], "y": [], "yaw": []}
        self._odom_poses = {"x": [], "y": []}
        self._sum_position_proposed = 0.0
        self._sum_orientation_proposed = 0.0
        self._sum_position_comparison = 0.0
        self._sum_orientation_comparison = 0.0
        self._proposed_errors_position = []
        self._proposed_errors_orientation = []
        self._comparison_errors_position = []
        self._comparison_errors_orientation = []
        self._evaluated = False

        self._gt_sub = message_filters.Subscriber("/amcl_pose", PoseWithCovarianceStamped)
        self._localized_pose_sub = message_filters.Subscriber("/vbatman/localized_pose", PoseStamped)
        self._observations_sub = message_filters.Subscriber("/vbatman/observations", PoseArray)
        self._reference_nodes_sub = message_filters.Subscriber("/vbatman/reference_nodes", NodeArray)
        self._current_index_sub = message_filters.Subscriber("/vbatman/current_index", Int32)
        self._odom_sub = message_filters.Subscriber("/whill/odom", Odometry)
        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            [self._gt_sub, self._localized_pose_sub, self._observations_sub,
             self._reference_nodes_sub, self._current_index_sub, self._odom_sub], 10, 0.1, allow_headerless=True)

        plt.rcParams["font.family"] = "Roboto"
        plt.rcParams["font.size"] = 15
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"
        plt.rcParams["axes.grid"] = True
        plt.rcParams["legend.fancybox"] = False
        plt.rcParams["legend.framealpha"] = 1.0
        plt.rcParams["legend.edgecolor"] = "black"

    def __call__(self) -> None:
        self._synchronizer.registerCallback(self._callback)
        rospy.spin()

    def __del__(self) -> None:
        if not self._evaluated:
            self.evaluate_localization()

    def _callback(self, gt_msg: PoseWithCovarianceStamped, localized_pose_msg: PoseStamped, observations_msg: PoseArray,
                  reference_nodes: NodeArray, current_index_msg: Int32, odom_msg: Odometry) -> None:
        self._count_steps += 1
        index = self._n_observations // 2
        if current_index_msg.data < self._n_observations // 2:
            index = current_index_msg.data
        node_pose = cast(List[Node], reference_nodes.nodes)[index].pose
        if self._count_steps == 1:
            self._init_odom = odom_msg.pose.pose

        gt = gt_msg.pose.pose
        proposed = localized_pose_msg.pose
        comparison = calc_transformed_pose(
            node_pose,
            cast(List[Pose], observations_msg.poses)[index])
        odom = odom_msg.pose.pose
        odom_pose = calc_transformed_pose(
            get_msg_from_array_2d(list(self._init_pose)),
            calc_relative_pose(self._init_odom, odom))

        self._gt_poses["x"].append(gt.position.x)
        self._gt_poses["y"].append(gt.position.y)
        self._gt_poses["yaw"].append(get_array_2d_from_msg(gt)[2])
        self._localized_poses["x"].append(proposed.position.x)
        self._localized_poses["y"].append(proposed.position.y)
        self._localized_poses["yaw"].append(get_array_2d_from_msg(proposed)[2])
        self._comparison_poses["x"].append(comparison.position.x)
        self._comparison_poses["y"].append(comparison.position.y)
        self._comparison_poses["yaw"].append(get_array_2d_from_msg(comparison)[2])
        self._odom_poses["x"].append(odom_pose.position.x)
        self._odom_poses["y"].append(odom_pose.position.y)

        relative_pose_proposed = get_array_2d_from_msg(
            calc_relative_pose(gt, proposed))
        relative_pose_comparison = get_array_2d_from_msg(
            calc_relative_pose(gt, comparison))
        self._sum_position_proposed += relative_pose_proposed[0] ** 2 + relative_pose_proposed[1] ** 2
        self._sum_orientation_proposed += math.degrees(relative_pose_proposed[2]) ** 2
        self._sum_position_comparison += relative_pose_comparison[0] ** 2 + relative_pose_comparison[1] ** 2
        self._sum_orientation_comparison += math.degrees(relative_pose_comparison[2]) ** 2

        if (current_index_msg.data == -1
                and np.linalg.norm(get_array_2d_from_msg(calc_relative_pose(gt, node_pose))[:2]) < 0.8):
            self.evaluate_localization()

    def plot_path(self) -> None:
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111, xlabel="x [m]", ylabel="y [m]")
        ax.set_xlim(-20, 35)

        ax.plot(self._gt_poses["x"], self._gt_poses["y"],
                marker=None, linestyle="-", label="ground truth")
        ax.plot(self._localized_poses["x"], self._localized_poses["y"],
                marker=None, linestyle="-.", label="VBATMAN")
        ax.plot(self._comparison_poses["x"], self._comparison_poses["y"],
                marker=None, linestyle="--", label="w/o filtering")
        ax.plot(self._odom_poses["x"], self._odom_poses["y"],
                marker=None, linestyle=":", label="odometry")
        ax.legend(loc="lower right")
        plt.savefig(f"{self._save_dir}/path.svg", bbox_inches="tight", pad_inches=0.05)
        plt.show()

    def plot_errors(self) -> None:
        fig_p = plt.figure(figsize=(8, 6))
        ax_p = fig_p.add_subplot(111, xlabel="Steps", ylabel="Error [m]")
        # ax_p.set_xlim(0, 80)
        ax_p.set_xlim(0, 140)
        ax_p.set_ylim(0, 2.5)

        ax_p.plot(self._proposed_errors_position, marker=None, linestyle="-", label="VBATMAN")
        ax_p.plot(self._comparison_errors_position, marker=None, linestyle="-.", label="w/o filtering")
        ax_p.legend(loc="upper right")
        plt.savefig(f"{self._save_dir}/errors_postion.svg", bbox_inches="tight", pad_inches=0.05)
        plt.show()

        fig_o = plt.figure(figsize=(8, 6))
        ax_o = fig_o.add_subplot(111, xlabel="Steps", ylabel="Error [deg]")
        # ax_o.set_xlim(0, 80)
        ax_o.set_xlim(0, 140)
        ax_o.set_ylim(0, 40)

        ax_o.plot(self._proposed_errors_orientation, marker=None, linestyle="-", label="VBATMAN")
        ax_o.plot(self._comparison_errors_orientation, marker=None, linestyle="-.", label="w/o filtering")
        ax_o.legend(loc="upper right")
        plt.savefig(f"{self._save_dir}/errors_orientation.svg", bbox_inches="tight", pad_inches=0.05)
        plt.show()

    def evaluate_localization(self) -> None:
        rmse_position_proposed = math.sqrt(self._sum_position_proposed / self._count_steps)
        rmse_orientation_proposed = math.sqrt(self._sum_orientation_proposed / self._count_steps)
        rmse_position_comparison = math.sqrt(self._sum_position_comparison / self._count_steps)
        rmse_orientation_comparison = math.sqrt(self._sum_orientation_comparison / self._count_steps)
        rospy.loginfo(f"[Proposed] position: {rmse_position_proposed}, orientation: {rmse_orientation_proposed}")
        rospy.loginfo(f"[Comparison] position: {rmse_position_comparison}, orientation: {rmse_orientation_comparison}")
        self.plot_path()

        self._proposed_errors_position = [
            math.hypot(gt_x - p_x, gt_y - p_y)
            for gt_x, gt_y, p_x, p_y
            in zip(self._gt_poses["x"], self._gt_poses["y"], self._localized_poses["x"], self._localized_poses["y"])
        ]
        self._proposed_errors_orientation = [
            abs(math.degrees(math.atan2(math.sin(gt_o - p_o), math.cos(gt_o - p_o))))
            for gt_o, p_o
            in zip(self._gt_poses["yaw"], self._localized_poses["yaw"])
        ]
        self._comparison_errors_position = [
            math.hypot(gt_x - c_x, gt_y - c_y)
            for gt_x, gt_y, c_x, c_y
            in zip(self._gt_poses["x"], self._gt_poses["y"], self._comparison_poses["x"], self._comparison_poses["y"])
        ]
        self._comparison_errors_orientation = [
            abs(math.degrees(math.atan2(math.sin(gt_o - c_o), math.cos(gt_o - c_o))))
            for gt_o, c_o
            in zip(self._gt_poses["yaw"], self._comparison_poses["yaw"])
        ]
        self.plot_errors()

        self._evaluated = True
