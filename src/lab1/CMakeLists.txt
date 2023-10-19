add_executable(test_lab1 lab1.cpp)
target_link_libraries(test_lab1 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)
