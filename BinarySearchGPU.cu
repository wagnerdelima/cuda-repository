###
##
/*
This program is an implementation of the Binary Search algorithm. 
This implementation uses CUDA in order to gain performance. It performs searches in O(1).
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <book.h>

#define N_ELEMENTS 2048

#define FLAG 0 //activates printing element

int BinarySearch(int *_array, int number_of_elements, int key);

__global__ void binarySearchGPU(int *arr, int key, int *ret){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(arr[tid] == key){
		ret[0] = tid;
	}
}
// inicializes the array with elements
__global__ void init(int *elem){

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < N_ELEMENTS){
		elem[tid] = tid;
	}
}

int main()
{
	//host array
	int *elem;
	//device array
	int *elem_d;
	//size of arrays
	size_t size = N_ELEMENTS * sizeof(int);
	//returned value
	int ret;
	int *ret_d;
	//seeking value
	int key = 0;

	//creating events to record elapsed time
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	dim3 blocks(2);
	dim3 threads(N_ELEMENTS/2);

	elem = (int*)malloc(size);
	HANDLE_ERROR(cudaMalloc((void**)&elem_d, size));
	HANDLE_ERROR(cudaMalloc((void**)&ret_d, sizeof(int)));

	printf("Please type the seeking value in a range of 0 to %i: \n", N_ELEMENTS);
	scanf("%i", &key);
	//start event
	cudaEventRecord(start, 0);

	init<<<blocks, threads>>>(elem_d);
	binarySearchGPU<<<2, 1024>>>(elem_d, key, ret_d);
	//stop event
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapsedtime;
	cudaEventElapsedTime(&elapsedtime, start, stop);

	HANDLE_ERROR(cudaMemcpy(elem, elem_d, size, cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&ret, ret_d, sizeof(int), cudaMemcpyDeviceToHost));

	printf("The element is in the position: %i\n", ret);
	printf("The algorithm has spent %f ms to find the key\n", elapsedtime);

	if(FLAG){
		for(int i = 0; i < N_ELEMENTS; i++){
			printf("\t%i ", elem[i]);
		
			if(i % 5 == 0)
				 printf("\n"); 
		}
		printf("\n");
	}

	//destroying events
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	free(elem);
	cudaFree(elem_d);
	cudaFree(ret_d);

	system("PAUSE");
    return 0;
}
