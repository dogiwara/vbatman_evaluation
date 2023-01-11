cmake_minimum_required(VERSION 3.0.2)
project(vbatman_evaluation)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O2")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0")

find_package(catkin REQUIRED COMPONENTS
  rospy
  geometry_msgs
)

catkin_package(
  LIBRARIES vbatman_evaluation
  CATKIN_DEPENDS rospy geometry_msgs vbatman
)
