#include <cpprest/http_listener.h>
#include <cpprest/filestream.h>
#include "image_processor.hpp"

using namespace web;
using namespace web::http;
using namespace web::http::experimental::listener;

void handle_options(http_request request) {
    http_response response(status_codes::OK);
    response.headers().add("Access-Control-Allow-Origin", "*");
    response.headers().add("Access-Control-Allow-Methods", "GET,POST,OPTIONS,PUT,DELETE,PATCH");
    response.headers().add("Access-Control-Allow-Headers", "Authorization,Accept,Origin,DNT,X-CustomHeader,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Content-Range,Range");
    request.reply(response);
}

void handle_post(http_request request) {
    http_response response(status_codes::OK);
    response.headers().add("Access-Control-Allow-Origin", "*");
    response.headers().add("Access-Control-Allow-Methods", "GET,POST,OPTIONS,PUT,DELETE,PATCH");
    response.headers().add("Access-Control-Allow-Headers", "Authorization,Accept,Origin,DNT,X-CustomHeader,Keep-Alive,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Content-Range,Range");

    auto content_type = request.headers().content_type();
    
    // Check if the content type is "image/jpeg" or similar
    if (content_type == "image/jpeg" || content_type == "image/png") {
        request.extract_vector().then([=](std::vector<unsigned char> body_data) {
            cv::Mat img = cv::imdecode(body_data, cv::IMREAD_COLOR);
            
            CustomImageProcessor process;
            cv::Mat processed_img = process.process_image(img);
            
            // Encode the result as JPEG
            std::vector<unsigned char> processed_data;
            cv::imencode(".jpg", processed_img, processed_data);

            // Return the processed image as a response
            http_response response(status_codes::OK);
            response.headers().add("Access-Control-Allow-Origin", "*");
            response.headers().add("Content-Type", "image/jpeg");
            response.set_body(processed_data);
            request.reply(response);
        });
    } else {
        request.reply(status_codes::UnsupportedMediaType);
    }

    request.reply(response);
}

int main() {
    http_listener listener(U("http://localhost:8080/image"));

    listener.support(methods::POST, handle_post);
    listener.support(methods::OPTIONS, handle_options);  // Support CORS preflight requests

    try {
        listener.open().wait();
        std::cout << "Server started at http://localhost:8080" << std::endl;
        std::string line;
        std::getline(std::cin, line); // Keep the server running
    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}
