add_executable(test_example example.cpp)
target_link_libraries(test_example PRIVATE doctest::doctest)
