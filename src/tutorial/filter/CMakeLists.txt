add_executable(test_filter filter.cpp)
target_link_libraries(test_filter 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)