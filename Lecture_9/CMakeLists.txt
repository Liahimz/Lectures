cmake_minimum_required(VERSION 3.10)

# Set project name
project(image_server)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find OpenCV package (dynamically link OpenCV)
find_package(OpenCV REQUIRED)
find_package(cpprestsdk REQUIRED)
find_package(OpenSSL REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(include)  

# Add executable
add_executable(${PROJECT_NAME} server.cpp)

# Add subdirectory for src
add_subdirectory(src)

# Link OpenCV (dynamic linking)
target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
target_link_libraries(${PROJECT_NAME} PRIVATE cpprestsdk::cpprest OpenSSL::SSL OpenSSL::Crypto)