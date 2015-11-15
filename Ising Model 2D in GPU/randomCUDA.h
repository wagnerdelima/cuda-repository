
/*
 * Ising Model 2D through Metropolis Algorithm
 *
 * This application implements the Metropolis algorithm applied to the 2D Ising Model. 
 */

/*
 * randomCUDA.h
 *
 *  Created on: Nov 20, 2013
 *      Author: Wagner de Lima
 */

#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cutil.h>

#define RANDOM_A 1664525
#define RANDOM_B 1013904223

#ifndef RANDOMCUDA_H_
#define RANDOMCUDA_H_

using namespace std;

//device which generates random numbers due a state and an integer
__device__ float generate(curandState* globalState, int ind) {

	//int ind = threadIdx.x;
	curandState localState = globalState[ind];
	float RANDOM = curand_uniform(&localState);
	globalState[ind] = localState;
	return RANDOM;

}

//establish a global state
__global__ void setup_kernel(curandState * state, unsigned long seed) {

	int id = threadIdx.x;
	curand_init(seed, id, 0, &state[id]);

}
//creates a few random numbers due states, a seek and an array to hold the numbers
__global__ void kernel(int* N, curandState* globalState, int n) {

	// generate random numbers
	//int i = threadIdx.x;

	int tx = threadIdx.x, ty = threadIdx.y, bx = blockIdx.x, by = blockIdx.y;
	//compute one dimentional matrix index
	int id = (by * blockDim.y + ty) * gridDim.x * blockDim.x + bx * blockDim.x + tx;


	int k = fabs(generate(globalState, tx) * RANDOM_B);

	while (k > n * n - 1) {

		k -= (n * n - 1);
	}

	N[id] = k;

}

typedef struct randomCUDA{

		float processingTime;

		//const int cudaCores = 64; // quantity of cores on the device
		//generates random numbers due an array and the number of threads
	__host__ void generateRandomNumbers(int* arrayNumbers,	int numberThreads){

		curandState* devStates; //creates random states for generating random numbers
		cudaMalloc(&devStates, sizeof(curandState));

		//creates two events for recording time
		cudaEvent_t start, end;
		const int cores = numberThreads / 8;

		cudaEventCreate(&start);
		cudaEventCreate(&end);

		cudaEventRecord(start, 0);

		dim3 threads(cores, cores);
		dim3 grid(numberThreads / 128, cores);

		// setup seeds
		setup_kernel<<<1, 200>>>(devStates, unsigned(time(NULL)));

		kernel<<<grid, threads>>>(arrayNumbers, devStates, RANDOM_A);

		cudaEventRecord(end, 0);
		cudaEventSynchronize(end);

		cudaEventElapsedTime(&processingTime, start, end);

	}

} randCUDA;

#endif /* RANDOMCUDA_H_ */
