# SPDX-License-Identifier: GPL-3.0-only
# Copyright (C) 2025-2026 Luo1imasi

cmake_minimum_required(VERSION 3.16)
project(roboparty_camera)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_COMPILER_LAUNCHER ccache)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_BUILD_TYPE Release)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc photo)

add_subdirectory(thirdparty)

add_executable(depth_node
  src/depth_node.cpp
  src/depth_provider.cpp
)
target_include_directories(depth_node
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
  ${ONNXRUNTIME_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(depth_node
  PUBLIC
  ${ONNXRUNTIME_LIB_FILE}
  ${OpenCV_LIBS}
  pthread
)
ament_target_dependencies(depth_node PUBLIC rclcpp std_msgs sensor_msgs std_srvs)
set_target_properties(depth_node PROPERTIES LINK_FLAGS "-Wl,--no-as-needed")

install(TARGETS depth_node
  RUNTIME DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY configs models launch
  DESTINATION share/${PROJECT_NAME})

ament_package()
