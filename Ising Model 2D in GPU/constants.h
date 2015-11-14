/*
 * constants.h
 *
 *  Created on: Nov 19, 2013
 *      Author: Wagner de Lima
 */

#ifndef CONSTANTS_H_
#define CONSTANTS_H_

#define TEMP_MIN 0.1 //minimum temperature
#define TEMP_MAX 4.1 //maximum temperature
#define GLOBAL_ITERATIONS 100

//linear congruential random numbers
#define RANDOM_A 1664525
#define RANDOM_B 1013904223


const unsigned int N = 4 * BLOCK_SIZE * BLOCK_SIZE;
const unsigned int n = 2 * BLOCK_SIZE;


#endif /* CONSTANTS_H_ */
