cmake_minimum_required(VERSION 3.10)
project(FeatureExtraction)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV CONFIG REQUIRED)

add_executable(SIFT SIFT.cpp)
target_link_libraries(SIFT PRIVATE ${OpenCV_LIBS})

add_executable(SURF SURF.cpp)
target_link_libraries(SURF PRIVATE ${OpenCV_LIBS})

add_executable(FAST FAST.cpp)
target_link_libraries(FAST PRIVATE ${OpenCV_LIBS})

add_executable(ORB ORB.cpp)
target_link_libraries(ORB PRIVATE ${OpenCV_LIBS})

## 用自己编译的（动态库），少了这部分
#foreach(target SIFT SURF FAST ORB)
#    add_custom_command(TARGET ${target} POST_BUILD
#            COMMAND ${CMAKE_COMMAND} -E copy_directory
#            "${OpenCV_DIR}/../bin"          # DLL 所在目录
#            $<TARGET_FILE_DIR:${target}>   # exe 输出目录
#    )
#endforeach()