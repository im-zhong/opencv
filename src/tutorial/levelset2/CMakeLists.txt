cmake_minimum_required(VERSION 3.22)
project( levelset2 )
find_package( OpenCV REQUIRED )

add_executable( levelset2 levelset2.cpp )
target_link_libraries( levelset2 ${OpenCV_LIBS} )
