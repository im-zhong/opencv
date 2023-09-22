cmake_minimum_required(VERSION 3.22)
project( images )
find_package( OpenCV REQUIRED )

add_executable( images images.cpp )
target_link_libraries( images ${OpenCV_LIBS} )
