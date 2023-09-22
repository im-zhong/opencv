cmake_minimum_required(VERSION 3.22)
project( filter )
find_package( OpenCV REQUIRED )

add_executable( filter filter.cpp )
target_link_libraries( filter ${OpenCV_LIBS} )
