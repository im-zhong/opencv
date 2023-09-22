project(lab1)

# static lab1 library
add_library(lab1 STATIC 
    src/lab1.cpp
)
target_include_directories(lab1 
    PUBLIC include
)
target_link_libraries(lab1 ${OpenCV_LIBS})

# lab1 unit tests
add_executable(test_lab1 
    test/test_lab1.cpp
)
target_link_libraries(test_lab1
    PRIVATE lab1
)
