# CMake file for the src directory

# Add the source files in this directory
set(SOURCES src/image_processor.cpp)

# Add these sources to the project executable
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
