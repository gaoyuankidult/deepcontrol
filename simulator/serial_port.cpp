#include "serial_port.h"


#define ASSERT(left,operator,right) { if(!((left) operator (right))){ std::cerr << "ASSERT FAILED: " << #left << #operator << #right << " @ " << __FILE__ << " (" << __LINE__ << "). " << #left << "=" << (left) << "; " << #right << "=" << (right) << std::endl; } }

SerialPort::SerialPort()
{
    this->context = zmq_ctx_new();
    this->requester = zmq_socket(context, ZMQ_PAIR);
    this->rc = zmq_connect(requester, "tcp://localhost:5556");
}

void SerialPort::send(std::string str, bool verbose){
    ASSERT(str.length(),<=,120);
    int rc = zmq_send(requester, str.c_str(), str.length(),0);
    if(verbose) {
        if(rc >0) {
            std::cout << "Successfully sent string:" << str << std::endl;
            std::cout << std::endl;
        }
        else {
            std::cerr << "Cant send information." << std::endl;
            std::cout << std::endl;
        }
    }

}

std::string SerialPort::receive(bool verbose){
    int num = zmq_recv(this->requester, buffer, 128, 0);
    std::string str(this->buffer);
    if(verbose){
        if(num > 0){
            std::cout << "Successfully received msg: " << str << std::endl;
            std::cout << std::endl;
        }
        else{
            std::cerr << "Cant receive information." << std::endl;
            std::cout << std::endl;
        }
    }

    return str;
}

SerialPort::~SerialPort()
{
    zmq_close(requester);
    zmq_ctx_destroy(context);

}

