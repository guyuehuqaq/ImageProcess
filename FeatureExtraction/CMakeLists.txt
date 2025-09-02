cmake_minimum_required(VERSION 3.10)
project(FeatureExtraction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV CONFIG REQUIRED)

add_executable(SIFT SIFT.cpp)
target_link_libraries(SIFT PRIVATE ${OpenCV_LIBS})

add_executable(SURF SURF.cpp)
target_link_libraries(SURF PRIVATE ${OpenCV_LIBS})