#include <iostream>
#include <unistd.h>
#include <cstring>
#include <sys/shm.h>
int main() {
    key_t key = ftok("shmfile", 65); // Generate a key
    int shmid = shmget(key, 1024, 0666 | IPC_CREAT); // Create a shared memory segment

    if (fork() == 0) {
        // Child process: writes data to the shared memory
        char *str = (char*) shmat(shmid, nullptr, 0);
        strcpy(str, "Hello from the child process!");
        std::cout << "Child process wrote data to shared memory\n";
        shmdt(str); // Detach the shared memory
    } else {
        sleep(1); // Wait for the child process to write the data
        // Parent process: reads data from the shared memory
        char *str = (char*) shmat(shmid, nullptr, 0);
        std::cout << "Parent process read from shared memory: " << str << std::endl;
        shmdt(str); // Detach the shared memory

        // Remove the shared memory segment after the work is done
        shmctl(shmid, IPC_RMID, nullptr);
    }
    return 0;
}
