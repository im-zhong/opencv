project(lab4)

# static lab4 library
add_executable(lab4
    src/lab4.cpp
)
target_include_directories(lab4
    PUBLIC include
)
target_link_libraries(lab4 ${OpenCV_LIBS})

# lab4 unit tests
# add_executable(test_lab4 
#     test/test_lab4.cpp
# )
# target_link_libraries(test_lab4
#     PRIVATE lab4
#     PRIVATE doctest::doctest
# )
