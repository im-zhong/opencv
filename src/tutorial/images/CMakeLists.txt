add_executable(test_images images.cpp)
target_link_libraries(test_images 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)