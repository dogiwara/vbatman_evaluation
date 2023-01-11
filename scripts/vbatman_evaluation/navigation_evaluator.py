import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, cast

import numpy as np
import rospy
from geometry_msgs.msg import Pose, PoseWithCovarianceStamped
from transformutils import calc_relative_pose, get_array_2d_from_msg
from vbatman.srv import (GetGTPoses, GetGTPosesResponse, ResetPath,
                         ResetPathResponse)


@dataclass(frozen=True)
class Config:
    hz: float
    n_trial: int
    start_dist_th: float
    goal_dist_th: float
    sleep_time: float
    timeout: float
    log_dir: str


class Status(Enum):
    INITIALIZING = auto()
    RUNNING = auto()


class NavigationEvaluator:
    _config: Config
    _cuurent_trial: int
    _status: Status
    _pose: Pose
    _start_gt_pose: Pose
    _goal_gt_pose: Pose
    _start_time: rospy.Time
    _results: Dict[str, int]
    _pose_sub: rospy.Subscriber
    _reset_path_client: rospy.ServiceProxy
    _get_gt_poses_client: rospy.ServiceProxy

    def __init__(self) -> None:
        rospy.init_node("navigation_evaluator")
        self._set_config()

        self._cuurent_trial = 0
        self._status = Status.INITIALIZING
        self._pose = Pose()
        self._start_gt_pose = Pose()
        self._goal_gt_pose = Pose()
        self._start_time = rospy.Time()
        self._results = {
            "success": 0,
            "init_failed": 0,
            "timeout": 0
        }

        self._pose_sub = rospy.Subscriber(
            "/amcl_pose", PoseWithCovarianceStamped, self._pose_callback)
        self._reset_path_client = rospy.ServiceProxy("/vbatman/reset_path", ResetPath)
        self._get_gt_poses_client = rospy.ServiceProxy("/vbatman/get_gt_poses", GetGTPoses)

        self.on_initialize_path()

    def __call__(self) -> None:
        rospy.Timer(
            rospy.Duration(nsecs=int(1.0 / 10.0 * 1e9)),
            self._timer_callback)
        rospy.spin()

    def _set_config(self) -> None:
        self._config = Config(
            rospy.get_param("~hz", 15.0),
            rospy.get_param("~n_trial", 100),
            rospy.get_param("~start_dist_th", 3.0),
            rospy.get_param("~goal_dist_th", 3.0),
            rospy.get_param("~sleep_time", 10),
            rospy.get_param("~timeout", 180),
            rospy.get_param("~log_dir", "/tmp")
        )

    def _timer_callback(self, _) -> None:
        if self._status is Status.INITIALIZING:
            return
        if rospy.Time.now() - self._start_time > rospy.Duration.from_sec(self._config.timeout):
            self._results["timeout"] += 1
            rospy.loginfo(self._results)
            self.on_initialize_path()
            return
        dist_to_goal = np.linalg.norm(
            get_array_2d_from_msg(calc_relative_pose(self._goal_gt_pose, self._pose))[:2])
        if dist_to_goal < self._config.goal_dist_th:
            self._results["success"] += 1
            rospy.loginfo(self._results)
            self.on_initialize_path()
            return

    def _pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._pose = msg.pose.pose

    def on_initialize_path(self) -> None:
        self._start_time = rospy.Time.now()
        self._status = Status.INITIALIZING
        self._cuurent_trial += 1
        if self._cuurent_trial > self._config.n_trial:
            self.on_finished()

        self.request_reset_path(-1)
        gt_poses = cast(List[Pose], self.request_get_gt_poses().gt_poses)
        self._start_gt_pose, self._goal_gt_pose = gt_poses[0], gt_poses[-1]
        dist_to_start = np.linalg.norm(
            get_array_2d_from_msg(calc_relative_pose(self._start_gt_pose, self._pose))[:2])

        if dist_to_start > self._config.start_dist_th:
            self._results["init_failed"] += 1
            rospy.loginfo(self._results)
            rospy.sleep(self._config.sleep_time)
            self.on_initialize_path()
        else:
            self._status = Status.RUNNING

    def on_finished(self) -> None:
        rospy.loginfo(f"NavigationEvaluator finished: {self._results}")
        log_file = datetime.now().strftime("%Y%m%d_%H%M.json")
        with open(f"{self._config.log_dir}/{log_file}", "w") as f:
            json.dump(self._results, f)

    def request_reset_path(self, target_id: int) -> ResetPathResponse:
        rospy.wait_for_service("/vbatman/reset_path")
        try:
            return self._reset_path_client(target_id)
        except rospy.ServiceException as e:
            rospy.logerr(f"Service /vbatman/reset_path call failed: {e}")
            return ResetPathResponse(False)

    def request_get_gt_poses(self) -> GetGTPosesResponse:
        rospy.wait_for_service("/vbatman/get_gt_poses")
        try:
            return self._get_gt_poses_client()
        except rospy.ServiceException as e:
            rospy.logerr(f"Service /vbatman/get_gt_poses call failed: {e}")
            return GetGTPosesResponse(False, [], [])
