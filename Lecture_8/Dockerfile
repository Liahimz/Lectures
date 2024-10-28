# Start with a CUDA base image
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Set environment variables to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update and install necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    wget \
    git \
    unzip \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Install CMake manually for the latest version
RUN wget -qO- https://cmake.org/files/v3.23/cmake-3.23.1-linux-x86_64.tar.gz | tar --strip-components=1 -xz -C /usr/local

# Set up project directory
WORKDIR /workspace

# Copy the project files into the container
COPY . .

# Configure CMake and build the project
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Run the compiled binary
CMD ["./build/mnist_convnet"]
