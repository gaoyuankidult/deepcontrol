#include <stdio.h>
#include <cmath>
#include <sstream>
#include <istream>
#include <vector>
#include <numeric>

#include "serial_port.h"
#include "random.h"

#define GRAVITY 9.8
#define MASSCART 1.0
#define MASSPOLE 0.1
#define TOTAL_MASS (MASSPOLE + MASSCART)
#define LENGTH 0.5		  /* actually half the pole's length */
#define POLEMASS_LENGTH (MASSPOLE * LENGTH)
#define FORCE_MAG 10.0
#define TAU 0.02		  /* seconds between state updates */
#define FOURTHIRDS 1.3333333333333
#define MAX_FAILURES     100000         /* Termination criterion. */
#define MAX_STEPS        100000

const bool VERBOSE = 0;


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

/*
 * Method Name	get_box:
 *
 * Synopsis	:
 *
 * This function checks whether the states of system is bounded in a range.
 * If the system state is not in a range, the na failture event is trigered.
 *
 * Parameters	:  float x, float x_dot, float theta, float theta_dot
 *
 * Description	: x is position of cart, x_dot is velocity of cart, theta is the angle between a perfectly balanced pole and theta_dot is velocity of theta
 *
 * Returns	: int box, box contains the information of which state does cart-pole system belongs to
 *
 * See Also	:
 */

#define OneDegree 0.0174532	/* 2pi/360 */
#define SixDegrees 0.1047192
#define TwelveDegrees 0.2094384
#define FiftyDegrees 0.87266

int get_box(float x, float x_dot, float theta, float theta_dot)
{
    int box=0;

    if (x < -2.4 ||
            x > 2.4  ||
            theta < -TwelveDegrees ||
            theta > TwelveDegrees)          return(-1); /* to signal failure */

    if (x < -0.8)  		       box = 0;
    else if (x < 0.8)     	       box = 1;
    else		    	               box = 2;

    if (x_dot < -0.5) 		       ;
    else if (x_dot < 0.5)                box += 3;
    else 			               box += 6;

    if (theta < -SixDegrees) 	       ;
    else if (theta < -OneDegree)        box += 9;
    else if (theta < 0) 		       box += 18;
    else if (theta < OneDegree) 	       box += 27;
    else if (theta < SixDegrees)        box += 36;
    else	    			       box += 45;

    if (theta_dot < -FiftyDegrees) 	;
    else if (theta_dot < FiftyDegrees)  box += 54;
    else                                 box += 108;

    return box;
}

void print_state(int action, float x, float x_dot, float theta, float theta_dot) {
    printf("The action given is: %d, \
           previous state of block is %f, \
           velocity of block is %f, \
           the angle is %f, \
           the velocity of angle is %f\n", \
           action, x, x_dot, theta, theta_dot);
}

int main(){

    // initialize communication class
    SerialPort serial = SerialPort();

    // receive numper of steps to unfold in time as int
    std::cout<< "waiting for the server to send the steps of unfolding in time..." << std::endl;
    std::string n_timesteps = serial.receive(VERBOSE);
    int n_times = std::stoi(n_timesteps);
    std::cout << "number of time steps received: " << n_times << std::endl;
    std::cout << std::endl;

    // initialize random related class
    Random rand = Random();


    // initialize parameters for training
    int action = 0;
    float raw_action = 0.0;
    float x = 0.0;
    float x_dot = 0.0;
    float theta = 0.0;
    float theta_dot = 0.0;
    int steps = 0;
    int failures = 0;
    int box = 0;
    int reward = 0;
    int best_steps = 0;
    std::ostringstream string_stream;
    std::vector<int> steps_list;
    float steps_average = 0.0;


    // iterate all training epoch.
    while (steps++ < MAX_STEPS && failures < MAX_FAILURES)
    {
        // ensure the training will start only if sequence is larger than backpropagation steps
        if(steps > n_times + 1) {
            //std::cout << "Now receiving the action from server..." << std::endl;
            raw_action = std::stof(serial.receive(VERBOSE));
            //std::cout << "raw action received is: " << raw_action << std::endl;
            action = raw_action < 0?-1:1;
        }
        // otherwise, choose action randomly
        else{
            // choose action between -1(move left), 0(stay still), 1(move right)
            action = rand.rand_int(-1,1);
        }

        // evaluate next state based on action and current state and ensure it is out of box
        cart_pole(action, &x, &x_dot, &theta, &theta_dot);
        box = get_box(x, x_dot, theta, theta_dot);

        // if out of box, then set reward for that step to be -100
        if (box < 0) {
            failures ++;
            reward = -7;
            x = x_dot = theta = theta_dot = 0.0;
            best_steps = steps>best_steps?steps:best_steps;
            steps_list.push_back(steps);
            steps = 0;
        }
        else{
            reward = -1;
        }

        // form information to be sent
        string_stream.str("");
        string_stream << reward <<"," << x <<"," << x_dot <<","<< theta <<","<< theta_dot << "\0";
        std::string str = string_stream.str();
        serial.send(str, VERBOSE);

        // receive the predicted action if states sequence given to the server are longer than n_time_steps
        if(0 == failures%10 && 0 == steps) {
            steps_average = std::accumulate(steps_list.begin(), steps_list.end(), 0)/float(steps_list.size());
            std::cout << "average steps: " << steps_average <<std::endl;
            std::cout << "best steps performed: " << best_steps << " currrent epoch is:" << failures << std::endl;
            steps_list.clear();
        }


    }
    if (failures == MAX_FAILURES)
        printf("Pole not balanced. Stopping after %d failures.",failures);
    else
        printf("Pole balanced successfully for at least %d steps\n", steps);







}
