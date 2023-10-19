add_executable(test_threshold threshold.cpp)
target_link_libraries(test_threshold 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)
