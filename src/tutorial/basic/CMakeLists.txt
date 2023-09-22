cmake_minimum_required(VERSION 3.22)
project( basic )
find_package( OpenCV REQUIRED )

add_executable( basic basic.cpp )
target_link_libraries( basic ${OpenCV_LIBS} )
