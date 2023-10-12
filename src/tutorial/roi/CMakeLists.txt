cmake_minimum_required(VERSION 3.22)
project(roi)
find_package(OpenCV REQUIRED)

add_executable(roi roi.cpp)
target_link_libraries(roi ${OpenCV_LIBS})
