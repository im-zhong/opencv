cmake_minimum_required(VERSION 3.22.0)
project(opencv CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

find_package(OpenCV REQUIRED)
find_package(doctest REQUIRED)

add_subdirectory(src)
