add_executable(test_roi roi.cpp)
target_link_libraries(test_roi 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)
