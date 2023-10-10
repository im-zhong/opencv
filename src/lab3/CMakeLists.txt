project(lab3)

# static lab3 library
add_executable(lab3 
    src/lab3.cpp
)
target_include_directories(lab3 
    PUBLIC include
)
target_link_libraries(lab3 ${OpenCV_LIBS})

# lab3 unit tests
# add_executable(test_lab3 
#     test/test_lab3.cpp
# )
# target_link_libraries(test_lab3
#     PRIVATE lab3
#     PRIVATE doctest::doctest
# )
