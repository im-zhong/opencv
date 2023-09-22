cmake_minimum_required(VERSION 3.22)
project(mask)
find_package(OpenCV REQUIRED)

add_executable(mask mask.cpp)
target_link_libraries(mask ${OpenCV_LIBS})
