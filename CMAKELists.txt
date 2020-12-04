cmake_minimum_required (VERSION 2.8.12)
project (RansacCircleDetection)

find_package(OpenCV REQUIRED)

add_executable(RansacCircleDetection main.cpp)
target_link_libraries (RansacCircleDetection 
  ${OpenCV_LIBS}
)
