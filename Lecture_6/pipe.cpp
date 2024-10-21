#include <iostream>
#include <unistd.h>
#include <cstring>

int main() {
    int pipefd[2];
    pid_t pid;
    char buffer[25];

    if (pipe(pipefd) == -1) {
        std::cerr << "Error creating channel\n";
        return 1;
    }

    pid = fork();
    if (pid == -1) {
        std::cerr << "Error creating process\n";
        return 1;
    }

    if (pid == 0) {
        // Child process writes data
        close(pipefd[0]); // Closing the unused side for reading
        const char* msg = "Hello from child process!\n";
        write(pipefd[1], msg, strlen(msg));
        close(pipefd[1]);
    } else {
        // Parent process reads data
        close(pipefd[1]); // Closing the unused side for recording
        read(pipefd[0], buffer, sizeof(buffer));
        std::cout << "The parent received a message: " << buffer << "\n";
        close(pipefd[0]);
    }

    return 0;
}