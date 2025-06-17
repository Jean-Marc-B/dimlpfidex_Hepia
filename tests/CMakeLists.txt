project(tests)

if(MSVC)
set(CMAKE_CXX_FLAGS "-O3 -fopenmp /EHsc")
else()
set(CMAKE_CXX_FLAGS "-O3 -fopenmp")
# set(CMAKE_CXX_FLAGS "-O3 -fopenmp -g -fsanitize=address -fsanitize=undefined")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -fopenmp -g -fsanitize=address -fsanitize=undefined")

add_subdirectory("cpp")

set_target_properties(
    tests PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/dimlpfidex
)

target_link_libraries(tests PUBLIC)
