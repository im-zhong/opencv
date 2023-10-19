add_executable(test_canny canny.cpp)
target_link_libraries(test_canny
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)