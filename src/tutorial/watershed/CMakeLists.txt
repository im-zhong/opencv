add_executable(test_watershed watershed.cpp)
target_link_libraries(test_watershed 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)
