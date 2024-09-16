cmake_minimum_required(VERSION 3.10)
project(YOLOv5_Inference)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Find Python3 package
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Get NumPy include directory
execute_process(COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
                OUTPUT_VARIABLE NumPy_INCLUDE_DIR
                OUTPUT_STRIP_TRAILING_WHITESPACE)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include directories
include_directories(
    ${Python3_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${NumPy_INCLUDE_DIR}  # Include the NumPy directory
)

# Set the path to your requirements.txt file
set(REQUIREMENTS_FILE "${CMAKE_SOURCE_DIR}/src/python/requirements.txt")

# Add a custom target to install Python packages using pip3
add_custom_target(
    install_python_packages
    COMMAND ${CMAKE_COMMAND} -E echo "Installing Python packages from ${REQUIREMENTS_FILE}..."
    COMMAND pip3 install -r ${REQUIREMENTS_FILE}
)

# Add a custom target to build the Python shared library
add_custom_target(build_python_shared_library
    COMMAND ${Python3_EXECUTABLE} src/python/setup.py build_ext --inplace
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Building Python shared library using setup.py"
)

# Add the executable
add_executable(run_inference src/main.cpp)

# Install Python packages for shared library
add_dependencies(run_inference install_python_packages)

# Make the executable dependent on the Python shared library
add_dependencies(run_inference build_python_shared_library)

# Link libraries
target_link_libraries(run_inference
    ${Python3_LIBRARIES}
    ${OpenCV_LIBS}
)

# Message outputs for confirmation
message(STATUS "Python3_INCLUDE_DIRS: ${Python3_INCLUDE_DIRS}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")


