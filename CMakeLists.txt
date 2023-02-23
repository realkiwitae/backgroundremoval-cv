cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
project(opencv-template) #name of your project

add_executable(myapp main.cpp utils.cpp)

find_package( OpenCV 3.2.0 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
target_compile_options(myapp PRIVATE -Wall -ggdb3)
target_link_libraries(myapp rt ${OpenCV_LIBS} pthread)
