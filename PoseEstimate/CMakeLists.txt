cmake_minimum_required(VERSION 3.10)
project(PoseEstimate)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /bigobj")

find_package(OpenCV CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(Ceres CONFIG REQUIRED)
find_package(g2o CONFIG REQUIRED)
find_package(Sophus CONFIG REQUIRED)

add_executable(PE_2d2d PE_2d2d.cpp)
target_link_libraries(PE_2d2d PRIVATE ${OpenCV_LIBS})

add_executable(PE_3d2d PE_3d2d.cpp)
target_link_libraries(PE_3d2d PRIVATE ${OpenCV_LIBS})

add_executable(PE_3d3d PE_3d3d.cpp
        PE_3d3d.h)
target_link_libraries(PE_3d3d PRIVATE
        ${OpenCV_LIBS}
        Eigen3::Eigen
        Ceres::ceres
        g2o::core
        Sophus::Sophus
)