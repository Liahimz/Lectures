# docker/Dockerfile.builder
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive
# Install dependencies
RUN apt-get update && \
    apt-get install -y cmake g++ libopencv-dev libssl-dev libcpprest-dev && \
    apt-get clean

# Set working directory
WORKDIR /app

# Copy source files to the container
COPY ../.. /app

# Run cmake and make to build the server
RUN cmake -S . -B build && cmake --build build
