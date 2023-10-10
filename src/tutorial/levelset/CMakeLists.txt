cmake_minimum_required(VERSION 3.22)
project( levelset )
find_package( OpenCV REQUIRED )

add_executable( levelset levelset.cpp )
target_link_libraries( levelset ${OpenCV_LIBS} )
