#include "serial_port.h"


SerialPort::SerialPort()
{
    this->context = zmq_ctx_new();
    this->requester = zmq_socket(context, ZMQ_PAIR);
    this->rc = zmq_connect(requester, "tcp://localhost:5556");
}

void SerialPort::send(std::string str){
    assert(str.length() <= 30);
    std::cout<< "Sender: Sending (%s)\n"<< str << std::endl;
    int rc = zmq_send(requester, str.c_str(), str.length(),0);
    //printf("check rc: %i\n", rc);
}

std::string SerialPort::receive(){

}

SerialPort::~SerialPort()
{
    zmq_close(requester);
    zmq_ctx_destroy(context);

}

