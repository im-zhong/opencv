cmake_minimum_required(VERSION 3.22)
project(opencv CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
set(CMAKE_EXPORT_COMPILE_COMMANDS on)

find_package(doctest REQUIRED)
find_package(OpenCV REQUIRED)

if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-DDOCTEST_CONFIG_DISABLE)
endif()

add_subdirectory(src)
