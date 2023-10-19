add_executable(test_mask mask.cpp)
target_link_libraries(test_mask 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)
