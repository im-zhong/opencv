add_executable(test_levelset2 levelset2.cpp)
target_link_libraries(test_levelset2 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)
