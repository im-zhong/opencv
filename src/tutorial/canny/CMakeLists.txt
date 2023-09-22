cmake_minimum_required(VERSION 3.22)
project( canny )
find_package( OpenCV REQUIRED )

add_executable( canny canny.cpp )
target_link_libraries( canny ${OpenCV_LIBS} )
