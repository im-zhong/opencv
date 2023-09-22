project(lab2)

# static lab2 library
add_library(lab2 STATIC 
    src/lab2.cpp
)
target_include_directories(lab2 
    PUBLIC include
)
target_link_libraries(lab2 ${OpenCV_LIBS})

# lab2 unit tests
add_executable(test_lab2 
    test/test_lab2.cpp
)
target_link_libraries(test_lab2
    PRIVATE lab2
)
