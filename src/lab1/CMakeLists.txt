project(lab1)

# static lab1 library
add_executable(lab1
    src/lab1.cpp
)
target_include_directories(lab1 
    PUBLIC include
)
target_link_libraries(lab1 ${OpenCV_LIBS})
