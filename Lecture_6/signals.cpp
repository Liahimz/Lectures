#include <iostream>
#include <unistd.h>
#include <signal.h>

// Signal handler
void signalHandler(int signum) {
    std::cout << "Signal received: " << signum << std::endl;
    exit(signum);
}

int main() {
    // Set the signal handler for SIGINT
    signal(SIGINT, signalHandler);

    if (fork() == 0) {
        // Child process waits for a signal
        std::cout << "Child process is waiting for SIGINT (Ctrl+C)...\n";
        while (true) {
            // Infinite loop, waiting for signals
        }
    } else {
        sleep(5); // Wait for a while
        std::cout << "Parent process is sending SIGINT to the child process\n";
        kill(0, SIGINT); // Send the signal to the child process
    }

    return 0;
}
