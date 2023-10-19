add_executable(test_levelset levelset.cpp)
target_link_libraries(test_levelset 
    PRIVATE ${OpenCV_LIBS}
    PRIVATE doctest::doctest
)
