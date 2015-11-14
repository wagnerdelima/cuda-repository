/*
 * calc.h
 *
 *  Created on: Nov 19, 2013
 *      Author: Wagner de Lima
 */
#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cutil.h>
#include "randomCUDA.h"
#include "device.h"

#ifndef CALC_H_
#define CALC_H_

#define TEMP_MIN 0.1 //minimum temperature
#define TEMP_MAX 4.1 //maximum temperature
#define GLOBAL_ITERATIONS 100
#define FLAG_PRINT_SPINS  0



__host__ void calc(int argc, char**argv) {
	printf(
			"----------------------------------------------------------------------- \n");
	printf(" *\n");
	printf(" * GPU accelerated Monte Carlo simulation of the 2D Ising model\n");
	printf(" *\n");
	printf(
			" * Copyright (C) 2013 LCCA - Developed by Wagner Leandro de Lima and Alberto E. P de Araujo \n");
	printf(" *\n");
	printf(" *\n");

	printf(" Number of Spins: %d \n", N);
	printf(" Start Temperature: %f \n", TEMP_MIN);
	printf(" Decreasing Factor: %f \n", TEMP_MIN);
	printf(" Final Temperature: %f \n", TEMP_MAX);
	printf(" Global Iterations: %d \n", GLOBAL_ITERATIONS);

	//Allocate and init host memory for simulation arrays
	unsigned int mem_size = sizeof(int) * N;
	//the random numbers lattice holds numbers to support the whole lattice of spins
	unsigned int mem_size_random = sizeof(int) * BLOCK_SIZE * BLOCK_SIZE;

	//host and device variables for random numbers and lattice of spins
	float allocatingTimeGpu = 0;
	int* h_random_data;
	int* d_random_data;
	int* dSpins;
	int* hSpins;
	unsigned int mem_size_out = sizeof(int) * BLOCK_SIZE;
	
	randCUDA createRandomNumbers;
	
	//recording events
	cudaEvent_t start, stop;

	//allocate memory for random numbers
	h_random_data = (int*) malloc(mem_size_random);
	//allocate memory for lattice of spins
	hSpins = (int*) malloc(mem_size);
	//**************************************************************************************************************************
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	//Start recording time of application
	cudaEventRecord(start, 0);

	//Create and allocate device memory for arrays
	CUDA_SAFE_CALL(cudaMalloc((void**) &d_random_data, mem_size_random));
	CUDA_SAFE_CALL(cudaMalloc((void**) &dSpins, mem_size));

	//Stop and destroy timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	//time spent in allocation
	float allocatingTime;
	cudaEventElapsedTime(&allocatingTime, start, stop);

	//Destroys events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	/*************************************************************************************************************************/
	//Build up the allocation time on gpu time
	allocatingTimeGpu += allocatingTime;

	printf("\n --------------------------------- GPU----------------------------------\n");
	printf(" Processing time on GPU for allocating: %f (ms) \n", allocatingTime);

	/************************************************************************************************************************/

	//generates an array of random numbers
	createRandomNumbers.generateRandomNumbers(d_random_data, BLOCK_SIZE);

	CUDA_SAFE_CALL(
			cudaMemcpy(h_random_data, d_random_data, mem_size_random, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaThreadSynchronize());

	//Build up time
	allocatingTimeGpu += createRandomNumbers.processingTime;

	printf("Time for initializing random numbers: %f (ms)\n",
			createRandomNumbers.processingTime);

	//************************************************************************************************************************

	//Create and start timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	dim3 threads(32, 32, 1);
	dim3 grid(16, 16, 1);

	//initializes the initial state
	inicializeSpinLattice<<<grid, threads>>>(dSpins, N);

	//Stop and destroy timer
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float initialStateTime;
	cudaEventElapsedTime(&initialStateTime, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	allocatingTimeGpu += initialStateTime;
	printf("Time for initializing initial state: %f (ms)\n", initialStateTime);
	//***********************************************************************************************************************

	dim3 cores(BLOCK_SIZE);
	dim3 blocks(BLOCK_SIZE);

	//Create and start timer
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start, 0);

	for (float temperature = TEMP_MIN; temperature < TEMP_MAX; temperature += 0.1) {
		for (int global_iteration = 0; global_iteration < GLOBAL_ITERATIONS;
				global_iteration++) {

			device_function_main<<<blocks, cores>>>(dSpins, d_random_data, temperature, true);
			device_function_main<<<blocks, cores>>>(dSpins, d_random_data, temperature, false);
			//CUDA_SAFE_CALL(cudaMemcpy(hSpins, dSpins, mem_size_out, cudaMemcpyDeviceToHost));
		}
	}
	CUDA_SAFE_CALL(cudaThreadSynchronize());
	
	//Stop and destroy timer
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float processingTimeGPU;
	cudaEventElapsedTime(&processingTimeGPU, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	allocatingTimeGpu += processingTimeGPU;
	
	printf(" Processing time on GPU for main function: %f (ms) \n", processingTimeGPU);
	printf(" Total processing time on GPU: %f (ms) \n", allocatingTimeGpu);
	//****************************************************************************************************************************************************************
	printf("-----------------------------------------------------------------------------\n");

	//free gpu memory
	cudaFree(d_random_data);
	cudaFree(dSpins);
	
	free(hSpins);
}

#endif /* CALC_H_ */

