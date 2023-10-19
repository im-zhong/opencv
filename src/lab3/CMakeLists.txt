add_executable(test_lab3 lab3.cpp)
target_link_libraries(test_lab3 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)
