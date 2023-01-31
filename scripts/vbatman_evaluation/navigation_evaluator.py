import pathlib
from dataclasses import dataclass
from enum import Enum, auto
from io import TextIOWrapper
from typing import List, cast

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
    log_path: str


class Status(Enum):
    INITIALIZING = auto()
    RUNNING = auto()


class NavigationEvaluator:
    _config: Config
    _cuurent_trial: int
    _log_file: TextIOWrapper
    _status: Status
    _pose: Pose
    _start_gt_pose: Pose
    _goal_gt_pose: Pose
    _start_time: rospy.Time
    _path_dist: float
    _pose_sub: rospy.Subscriber
    _reset_path_client: rospy.ServiceProxy
    _get_gt_poses_client: rospy.ServiceProxy

    def __init__(self) -> None:
        rospy.init_node("navigation_evaluator")
        self._set_config()

        log_path = pathlib.Path(self._config.log_path)
        if not log_path.parent.exists():
            log_path.parent.mkdir(parents=True)
        if not log_path.exists():
            log_path.touch()

        self._cuurent_trial = len(log_path.open("r").readlines())
        self._log_file = log_path.open("a")
        self._status = Status.INITIALIZING
        self._pose = Pose()
        self._start_gt_pose = Pose()
        self._goal_gt_pose = Pose()
        self._start_time = rospy.Time()
        self._path_dist = 0.0

        self._pose_sub = rospy.Subscriber(
            "/amcl_pose", PoseWithCovarianceStamped, self._pose_callback)
        self._reset_path_client = rospy.ServiceProxy("/vbatman/reset_path", ResetPath)
        self._get_gt_poses_client = rospy.ServiceProxy("/vbatman/get_gt_poses", GetGTPoses)

        self.on_initialize_path()

    def __del__(self) -> None:
        self._log_file.close()

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
            rospy.get_param("~log_path", "/tmp")
        )

    def _timer_callback(self, _) -> None:
        if self._status is Status.INITIALIZING:
            return
        if rospy.Time.now() - self._start_time > rospy.Duration.from_sec(self._config.timeout):
            self.write_log(False)
            self.on_initialize_path()
            return
        dist_to_goal = np.linalg.norm(
            get_array_2d_from_msg(calc_relative_pose(self._goal_gt_pose, self._pose))[:2])
        if dist_to_goal < self._config.goal_dist_th:
            self.write_log(True)
            self.on_initialize_path()
            return

    def _pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._pose = msg.pose.pose

    def on_initialize_path(self) -> None:
        self._start_time = rospy.Time.now()
        self._status = Status.INITIALIZING
        self._cuurent_trial += 1
        if self._cuurent_trial > self._config.n_trial:
            rospy.loginfo("Finished all trials")
            return

        self.request_reset_path(-1)
        gt_poses = cast(List[Pose], self.request_get_gt_poses().gt_poses)
        self._start_gt_pose, self._goal_gt_pose = gt_poses[0], gt_poses[-1]
        dist_to_start = np.linalg.norm(
            get_array_2d_from_msg(calc_relative_pose(self._start_gt_pose, self._pose))[:2])
        self._path_dist = np.linalg.norm(  # type: ignore
            get_array_2d_from_msg(calc_relative_pose(self._start_gt_pose, self._goal_gt_pose))[:2])

        if dist_to_start > self._config.start_dist_th:
            self.write_log(False)
            rospy.sleep(self._config.sleep_time)
            self.on_initialize_path()
        else:
            self._status = Status.RUNNING

    def write_log(self, success: bool) -> None:
        rospy.loginfo(f"Trial #{self._cuurent_trial}: dist={self._path_dist}, success={success}")
        self._log_file.write(f"{self._path_dist}, {success}\n")

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
