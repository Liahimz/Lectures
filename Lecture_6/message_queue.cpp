#include <iostream>
#include <unistd.h>
#include <cstring>
#include <sys/msg.h>
struct Message {
    long messageType;
    char messageText[100];
};
int main() {
    key_t key = ftok("queuefile", 65); // Create a unique key
    int msgid = msgget(key, 0666 | IPC_CREAT); // Create a message queue

    Message msg;
    msg.messageType = 1;
    if (fork() == 0) {
        // Child process: send message
        strcpy(msg.messageText, "Hello from child process!");
        msgsnd(msgid, &msg, sizeof(msg.messageText), 0);
        std::cout << "Message sent by child process\n";
    } else {
        // Parent process: receive message
        msgrcv(msgid, &msg, sizeof(msg.messageText), 1, 0);
        std::cout << "Message from child process: " << msg.messageText << std::endl;
        // Delete the message queue after receiving
        msgctl(msgid, IPC_RMID, nullptr);
    }
    return 0;
}
