#!/usr/bin/env python3

import rospy

from vbatman_evaluation import LocalizationEvaluator


def main() -> None:
    try:
        LocalizationEvaluator()()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
