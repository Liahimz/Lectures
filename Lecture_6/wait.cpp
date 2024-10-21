#include <iostream>
#include <unistd.h>
#include <sys/wait.h>

int main() {
    pid_t pid = fork();

    if (pid == 0) {
        std::cout << "Child process: running...\n";
        sleep(2);
        std::cout << "Child process: finished\n";
        return 0;
    } else if (pid > 0) {
        std::cout << "Parent process: waiting for the child process to finish...\n";
        wait(nullptr); // Waiting for the child process to finish
        std::cout << "Parent process: child process finished\n";
    }

    return 0;
}
