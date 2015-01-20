#ifndef SERIAL_PORT_H
#define SERIAL_PORT_H
#include <iostream>
#include <assert.h>
#include "zmq.h"

class SerialPort
{
private:
    char buffer[32];
    void *context;
    void *requester;
    int rc;
public:
    SerialPort();
    ~SerialPort();
    void send(std::string buffer);
    std::string receive();

};

#endif // SERIAL_PORT_H
