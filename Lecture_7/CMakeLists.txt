cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(MNISTConvNet LANGUAGES CXX CUDA)

# Enable C++11 standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable CUDA
find_package(CUDA REQUIRED)

# OpenCV package
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(mnist_convnet src/simple_convnet.cu)
add_executable(matrix_mul src/matrix_mul.cu)
add_executable(vector_add src/vector_add.cu)

# Link CUDA runtime library and OpenCV libraries
target_link_libraries(mnist_convnet ${CUDA_LIBRARIES} ${OpenCV_LIBS})
# # Link CUDA runtime library directly
target_link_libraries(vector_add ${CUDA_LIBRARIES})
target_link_libraries(matrix_mul ${CUDA_LIBRARIES})

# Include the necessary CUDA directories
include_directories(${CUDA_INCLUDE_DIRS})





