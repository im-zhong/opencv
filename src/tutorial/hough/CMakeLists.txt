add_executable(test_hough hough.cpp)
target_link_libraries(test_hough 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)

