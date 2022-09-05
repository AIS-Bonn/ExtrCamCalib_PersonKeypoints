cmake_minimum_required(VERSION 2.8.3)
project(keypoint_camera_calibration)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  message_generation
  person_msgs
)

add_message_files(
  FILES
  Person2DWithID.msg
)

generate_messages(
   DEPENDENCIES
   person_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)