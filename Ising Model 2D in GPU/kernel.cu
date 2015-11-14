
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <math.h>
#include <time.h>

#include "calc.h"

/*
 *s
 * Function Declaration
 *
 * */
__host__ void calc(int argc, char** argv);
__host__ void cpu_function(double*, int*);
__global__ void device_function_main(int*, int*, int*, float, bool);

/*
 * Main Function
 * */
int main(int argc, char ** argv) {
	
	calc(argc, argv);
	getchar();
	return 0;
}
