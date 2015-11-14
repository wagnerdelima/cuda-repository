
/*
 * device.h
 *
 *  Created on: Nov 19, 2013
 *      Author: Waner de Lima
 */

#ifndef DEVICE_H_
#define DEVICE_H_

#define TEMP_MIN 0.1 //minimum temperature
#define TEMP_MAX 4.1 //maximum temperature
#define GLOBAL_ITERATIONS 100

//linear congruential random numbers
#define RANDOM_A 1664525
#define RANDOM_B 1013904223
#define BLOCK_SIZE 256

const unsigned int N = 4 * BLOCK_SIZE * BLOCK_SIZE;
const unsigned int n = 2 * BLOCK_SIZE;

//due a lattice of spins and a size, creates an initial state
__global__ void inicializeSpinLattice(int *redeSpin, int tamanho) {

	int tx = threadIdx.x,
		ty = threadIdx.y,
		bx = blockIdx.x,
		by = blockIdx.y;
	//compute one dimentional matrix index
	int id = (by * blockDim.y + ty) * gridDim.x * blockDim.x + bx * blockDim.x
			+ tx;

	//int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < tamanho) {
		redeSpin[id] = 1;
	}
}

__global__ void device_function_main(int* S, int* R, float t,
		bool flag) {

	int sum_spins = 0;
	//Energy variable
	int dH = 0;
	float exp_dH_4 = exp(-(4.0) / t);
	float exp_dH_8 = exp(-(8.0) / t);

	//array which holds random numbers a possibly will hold magnetization numbers
	__shared__ int _array[BLOCK_SIZE];
	__shared__ int spins[BLOCK_SIZE * BLOCK_SIZE];
	//Load random data
	_array[threadIdx.x] = R[threadIdx.x + BLOCK_SIZE * blockIdx.x];
	__syncthreads();

	if (flag) {
	//Create new random numbers
		_array[threadIdx.x] = RANDOM_A * _array[threadIdx.x] + RANDOM_B;
			
	//Spin update top left
	if (blockIdx.x == 0) { //Top - it means that block 0 is the first line at a matrix of spins
		if (threadIdx.x == 0) { //Left
			sum_spins =  (S[2 * threadIdx.x + 1] //right
				+ S[2 * threadIdx.x - 1 + 2 * BLOCK_SIZE] //left
				+ S[2 * threadIdx.x + 2 * BLOCK_SIZE] //bottom
				+ S[2 * threadIdx.x + N - 2 * BLOCK_SIZE]); //top
			dH = S[2 * threadIdx.x] * sum_spins;
		} else {
			sum_spins = (S[2 * threadIdx.x + 1] + //right
							S[2 * threadIdx.x - 1] + //left
							S[2 * threadIdx.x + 2 * BLOCK_SIZE] + //bottom
							S[2 * threadIdx.x + N - 2 * BLOCK_SIZE]); //top
			dH = S[2 * threadIdx.x] * sum_spins; 
		}
		} else {
			if (threadIdx.x == 0) { //Left
				sum_spins =  (S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 1]
					+ S[2 * threadIdx.x	+ 4 * BLOCK_SIZE * blockIdx.x - 1 + 2 * BLOCK_SIZE] //left
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] //bottom
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x - 2 * BLOCK_SIZE]); //top
				
						dH = S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x] * sum_spins;
				} else {
					sum_spins =  (S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 1] //right
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x - 1] //left
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] //bottom
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x - 2 * BLOCK_SIZE]); //top
						
						dH = 2 * S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x] * sum_spins;
				}
			}
	
		if (dH == 4) {
			if (fabs(_array[threadIdx.x] * 4.656612e-10) < exp_dH_4) {
				S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x] = -S[2
						* threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x];
			}
		} else if (dH == 8) {
			if (fabs(_array[threadIdx.x] * 4.656612e-10) < exp_dH_8) {
				S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x] = -S[2
						* threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x];
			}
		} else {
			S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x] = -S[2
					* threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x];
		}

	__syncthreads();		
	//Create new random numbers
	//_array[threadIdx.x] = RANDOM_A * _array[threadIdx.x] + RANDOM_B;
	//Spin update bottom right
	if (blockIdx.x == BLOCK_SIZE - 1) { //Bottom
		if (threadIdx.x == BLOCK_SIZE - 1) { //Right
			sum_spins =  (S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2] //right
			+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] //left
			+ S[2 * threadIdx.x + 1] //bottom
			+ S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x]); //top
			
			dH = S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] * sum_spins;
			} else {
				sum_spins =  (S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE + 2] //right
				+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] //left
				+ S[2 * threadIdx.x + 1] //bottom
				+ S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x]); //top
				dH = 2 * S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] * sum_spins;
				}
			} else {
				if (threadIdx.x == BLOCK_SIZE - 1) { //Right
					sum_spins =  (S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2] //right
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] //left
						+ S[2 * threadIdx.x + 1+ 4 * BLOCK_SIZE * (blockIdx.x + 1)] //bottom
						+ S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x]); //top
						dH = 2 * S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] * sum_spins;
				} else {
					sum_spins =  (S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE + 2] //right
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] //left
						+ S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * (blockIdx.x + 1)] //bottom
						+ S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x]); //top
						dH = 2 * S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] * sum_spins;
				}
			}
	if (dH == 4) {
		if (fabs(_array[threadIdx.x] * 4.656612e-10) < exp_dH_4) {
				S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x
						+ 2 * BLOCK_SIZE] = -S[2 * threadIdx.x + 1
						+ 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE];
			}
		} else if (dH == 8) {
			if (fabs(_array[threadIdx.x] * 4.656612e-10) < exp_dH_8) {
				S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x
						+ 2 * BLOCK_SIZE] = -S[2 * threadIdx.x + 1
						+ 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE];
			}
		} else {
			S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] =
					-S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x
							+ 2 * BLOCK_SIZE];
		}
	__syncthreads();

	} else {
		//Create new random numbers
		_array[threadIdx.x] = RANDOM_A * _array[threadIdx.x] + RANDOM_B;
		//Spin update top right
		if (blockIdx.x == 0) { //Top
			if (threadIdx.x == BLOCK_SIZE - 1) { //Right
				sum_spins =  (S[2 * threadIdx.x + 2 - 2 * BLOCK_SIZE]
					+ S[2 * threadIdx.x]
					+ S[2 * threadIdx.x + 1 + 2 * BLOCK_SIZE]
					+ S[2 * threadIdx.x + 1 + N - 2 * BLOCK_SIZE]);
					dH = 2 * S[2 * threadIdx.x + 1] * sum_spins;
				} else {
					sum_spins = S[2 * threadIdx.x + 2]
					   + S[2 * threadIdx.x]
						+ S[2 * threadIdx.x + 1 + 2 * BLOCK_SIZE]
						+ S[2 * threadIdx.x + 1 + N - 2 * BLOCK_SIZE];
						dH = 2 * S[2 * threadIdx.x + 1] * sum_spins;
				}
			} else {
				if (threadIdx.x == BLOCK_SIZE - 1) { //Right
					sum_spins = S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 - 2 * BLOCK_SIZE]
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x]
						+ S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE]
						+ S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x - 2 * BLOCK_SIZE];
						dH = 2 * S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x] * sum_spins;
				} else {
					sum_spins = S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2]
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x]
						+ S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE]
						+ S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x - 2 * BLOCK_SIZE];
						dH = 2 * S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x] * sum_spins;
				}
			}
		if (dH == 4) {
			if (fabs(_array[threadIdx.x] * 4.656612e-10) < exp_dH_4) {
				S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x] = -S[2
						* threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x];
			}
		} else if (dH == 8) {
			if (fabs(_array[threadIdx.x] * 4.656612e-10) < exp_dH_8) {
				S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x] = -S[2
						* threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x];
			}
		} else {
			S[2 * threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x] = -S[2
					* threadIdx.x + 1 + 4 * BLOCK_SIZE * blockIdx.x];
		}

		__syncthreads();
		
		//Create new random numbers
		//_array[threadIdx.x] = RANDOM_A * _array[threadIdx.x] + RANDOM_B;
		//Spin update bottom left
		if (blockIdx.x == BLOCK_SIZE - 1) { //Bottom
			if (threadIdx.x == 0) { //Left
				sum_spins =  (S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE + 1]
					+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * (blockIdx.x + 1) - 1]
					+ S[2 * threadIdx.x]
					+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x]);
					dH = 2 * S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] * sum_spins;
				} else {
					sum_spins =  (S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE + 1]
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE - 1]
						+ S[2 * threadIdx.x]
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x]);
						dH = 2 * S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] * sum_spins;
				}
			} else {
				if (threadIdx.x == 0) { //Left
					sum_spins = (S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE + 1]
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * (blockIdx.x + 1) - 1]
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * (blockIdx.x + 1)]
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x]);
						dH = 2 * S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] * sum_spins;
				} else {
					sum_spins = (S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE + 1]
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE - 1]
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * (blockIdx.x + 1)]
						+ S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x]);
						dH = 2 * S[2 * threadIdx.x + 4 * BLOCK_SIZE * blockIdx.x + 2 * BLOCK_SIZE] * sum_spins;
				}
			}
		if (dH == 4) {
			if (fabs(_array[threadIdx.x] * 4.656612e-10) < exp_dH_4) {
				S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE] = -S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE];
			}
		} else if (dH == 8) {
			if (fabs(_array[threadIdx.x] * 4.656612e-10) < exp_dH_8) {
				S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE]=-S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE];
			}
		} else {
			S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE]=-S[2*threadIdx.x+4*BLOCK_SIZE*blockIdx.x+2*BLOCK_SIZE];
		}
		__syncthreads();

		}
}

#endif /* DEVICE_H_ */
