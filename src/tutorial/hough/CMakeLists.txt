cmake_minimum_required(VERSION 3.22)
project( hough )
find_package( OpenCV REQUIRED )

add_executable( hough hough.cpp )
target_link_libraries( hough ${OpenCV_LIBS} )
