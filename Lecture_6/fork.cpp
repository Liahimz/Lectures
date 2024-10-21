#include <iostream>
#include <unistd.h>

int main() {
    pid_t pid = fork();

    if (pid == -1) {
        std::cerr << "Error creating process\n";
        return 1;
    } else if (pid == 0) {
        // Child process code
        std::cout << "This is a child process!\n";
    } else {
        // Parent process code
        std::cout << "This is the parent process. Child process PID:" << pid << "\n";
    }

    return 0;
}
