from dataclasses import dataclass
from typing import Optional, cast

import numpy as np
import rosbag
import rospy
from geometry_msgs.msg import Pose
from sensor_msgs.msg import CompressedImage
from transformutils import calc_relative_pose, get_array_2d_from_msg
from vbatman.msg import Node, NodeArray
from vbatman.srv import GetPath, GetPathResponse


@dataclass(frozen=True)
class Config:
    image_width: int
    image_height: int
    bagfile_path: str
    image_topic_name: str
    pose_topic_name: str
    interval_r: float
    interval_yaw: float


class DummyPathPlanner:
    _config: Config
    _nodes: NodeArray
    _get_path_service: rospy.Service

    def __init__(self) -> None:
        rospy.init_node("dummy_path_planner")

        self._config = Config(
            cast(int, rospy.get_param("/vbatman/common/image_width")),
            cast(int, rospy.get_param("/vbatman/common/image_height")),
            cast(str, rospy.get_param("~bagfile_path")),
            cast(str, rospy.get_param("~image_topic_name")),
            cast(str, rospy.get_param("~pose_topic_name")),
            cast(float, rospy.get_param("~interval_r")),
            cast(float, rospy.get_param("~interval_yaw"))
        )
        self._nodes = NodeArray()
        self._generate_path(rosbag.Bag(self._config.bagfile_path))

        self._get_path_service = rospy.Service(
            "/vbatman/get_path", GetPath, self._handle_get_path)

    def __call__(self) -> None:
        rospy.spin()

    def _handle_get_path(self, _) -> GetPathResponse:
        return GetPathResponse(self._nodes)

    def _generate_path(self, bag: rosbag.Bag) -> None:
        topics = [self._config.image_topic_name, self._config.pose_topic_name]
        pose: Optional[Pose] = None
        prev_pose: Optional[Pose] = None
        image: Optional[CompressedImage] = None

        for topic, msg, _ in bag.read_messages(topics=topics):  # type: ignore
            if topic == self._config.image_topic_name:
                image = cast(CompressedImage, msg)
            elif topic == self._config.pose_topic_name:
                pose = cast(Pose, msg.pose.pose)
            if pose is not None and image is not None:
                if prev_pose is None:
                    self._nodes.nodes.append(  # type: ignore
                        Node(image, pose, pose))
                    prev_pose = pose
                    continue
                relative_pose = get_array_2d_from_msg(
                    calc_relative_pose(prev_pose, pose))
                if (np.linalg.norm(relative_pose[:2]) >= self._config.interval_r
                        or abs(relative_pose[2]) >= self._config.interval_yaw):
                    self._nodes.nodes.append(  # type: ignore
                        Node(image, pose, pose))
                    prev_pose = pose
                    image = None
                    pose = None
