cmake_minimum_required(VERSION 3.27)
project(cudafractals CUDA)

set(CMAKE_CUDA_STANDARD 17)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

add_executable(cudafractals main.cu)

target_link_libraries(cudafractals PRIVATE OpenGL::GL GLUT::GLUT GLEW::GLEW)

set_target_properties(cudafractals PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
)
