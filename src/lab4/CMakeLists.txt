add_executable(test_lab4 lab4.cpp)
target_link_libraries(test_lab4 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)
