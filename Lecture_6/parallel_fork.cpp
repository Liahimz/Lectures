#include <iostream>
#include <unistd.h>
#include <sys/wait.h>

void performTask(const std::string& taskName) {
    std::cout << "Process is performing task: " << taskName << "\n";
    sleep(2);
    std::cout << "Task " << taskName << " completed\n";
}

int main() {
    for (int i = 0; i < 3; ++i) {
        pid_t pid = fork();
        if (pid == 0) {
            // Child process performs its task
            performTask("Task " + std::to_string(i + 1));
            return 0;
        }
    }

    for (int i = 0; i < 3; ++i) {
        wait(nullptr); // Wait for all child processes to complete
    }

    std::cout << "All tasks are completed\n";
    return 0;
}
