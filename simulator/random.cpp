#include "random.h"

Random::Random()
{
    srand (time(NULL));
}

Random::~Random()
{

}

int Random::rand_int(int low, int high){
    // Random number between low and high
    return rand() % ((high + 1) - low) + low;
}

