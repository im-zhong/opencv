add_executable(test_basic basic.cpp)
target_link_libraries(test_basic 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)
