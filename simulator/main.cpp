#include <stdio.h>
#include <cmath>
#include <sstream>

#include "serial_port.h"

#define GRAVITY 9.8
#define MASSCART 1.0
#define MASSPOLE 0.1
#define TOTAL_MASS (MASSPOLE + MASSCART)
#define LENGTH 0.5		  /* actually half the pole's length */
#define POLEMASS_LENGTH (MASSPOLE * LENGTH)
#define FORCE_MAG 10.0
#define TAU 0.02		  /* seconds between state updates */
#define FOURTHIRDS 1.3333333333333


void cart_pole(int action,float* x, float* x_dot, float *theta, float* theta_dot)
{
    float xacc,thetaacc,force,costheta,sintheta,temp;

    force = (action>0)? FORCE_MAG : -FORCE_MAG;
    costheta = cos(*theta);
    sintheta = sin(*theta);

    temp = (force + POLEMASS_LENGTH * *theta_dot * *theta_dot * sintheta)
                 / TOTAL_MASS;

    thetaacc = (GRAVITY * sintheta - costheta* temp)
           / (LENGTH * (FOURTHIRDS - MASSPOLE * costheta * costheta
                                              / TOTAL_MASS));

    xacc  = temp - POLEMASS_LENGTH * thetaacc* costheta / TOTAL_MASS;

    /*** Update the four state variables, using Euler's method. ***/
    *x  += TAU * *x_dot;
    *x_dot += TAU * xacc;
    *theta += TAU * *theta_dot;
    *theta_dot += TAU * thetaacc;
}

int main(){
    int y = 1;
    float x = 0.1;
    float x_dot = 0.1;
    float theta = 0.1;
    float theta_dot = 0.5;
    printf("The action given is: %d, \
            previous state of block is %f, \
            velocity of block is %f, \
            the angle is %f, \
            the velocity of angle is %f\n", \
           y, x, x_dot, theta, theta_dot);
    cart_pole(y, &x, &x_dot, &theta, &theta_dot);
    printf("current state of block is %f, \
            velocity of block is %f, \
            the angle is %f, \
            the velocity of angle is %f", \
           y, x, x_dot, theta, theta_dot);

    std::ostringstream string_stream;
    string_stream << x <<"," << x_dot <<","<< theta <<","<< theta_dot << "\0";
    std::string c_str = string_stream.str();


    SerialPort serial = SerialPort();
    serial.send(c_str);
}
