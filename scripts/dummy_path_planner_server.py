#!/usr/bin/env python3

import rospy

from vbatman_evaluation import DummyPathPlanner


def main() -> None:
    try:
        DummyPathPlanner()()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
