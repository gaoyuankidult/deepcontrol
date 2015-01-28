#ifndef RANDOM_H
#define RANDOM_H

#include <time.h>
#include <stdlib.h>


class Random
{
public:
    Random();
    ~Random();

    int rand_int(int low, int high);
};

#endif // RANDOM_H
