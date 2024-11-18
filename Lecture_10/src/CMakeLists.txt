add_library(image_processor image_processor.cpp)
target_link_libraries(image_processor ${OpenCV_LIBS})

add_executable(main main.cpp)
target_link_libraries(main image_processor ${OpenCV_LIBS})
