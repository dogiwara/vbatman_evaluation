#!/usr/bin/env python3

import rospy

from vbatman_evaluation import NavigationEvaluator


def main() -> None:
    try:
        NavigationEvaluator()()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
