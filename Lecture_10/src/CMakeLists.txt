add_library(image_processor image_processor.cpp)
target_link_libraries(image_processor ${OpenCV_LIBS})

add_executable(process main.cpp)
target_link_libraries(process image_processor ${OpenCV_LIBS})
