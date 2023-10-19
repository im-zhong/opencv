add_executable(test_lab2 lab2.cpp)
target_link_libraries(test_lab2 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)