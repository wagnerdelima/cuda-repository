# Ising Model 2D through Metropolis Algorithm

This application implements the Metropolis algorithm applied to the 2D Ising Model. This code was adapted from Preis et. al. who have come up with a simple application whose main objective was to give the speed-up comparison of a CPU-based and GPU-based codes. As so this. However, I use more GPU power best speed up the algorithm. 

The initial state, initializing all spins on the square lattice is performed into GPU, differently from the literature, which does it through CPU. I have also created an API (found in the randomCUDA.h file header) the generate random numbers totally in GPU parallel power, and also calculate its elapsed time.

I deeply hope this is somehow useful to you. 

for contacting me:

Gmail: waglds@gmail.com
Skype: waglds

Thank you very much.
Muito obrigado.

#The Ising Model

This is an implementation of the 2D Ising Model using the Metropolis Algorithm. The Ising model consists of discrete variables that represent magnetic dipole moments of atomic spins that can be in one of two states (+1 or −1). You may find further details about the model in https://en.wikipedia.org/wiki/Ising_model.

#The Metropolis Algorithm
The Metropolis algorithm is a method of statistics and statistical physics, the Metropolis–Hastings algorithm is a Markov chain Monte Carlo (MCMC) method for obtaining a sequence of random samples from a probability distribution for which direct sampling is difficult. You may also find more details in https://en.wikipedia.org/wiki/Metropolis%E2%80%93Hastings_algorithm


#References
I based myself and adapted an application presented by Preis et. al.

GPU accelerated Monte Carlo simulation of the 2D and 3D Ising model
http://www.sciencedirect.com/science/article/pii/S0021999109001387

You may also find an extented abstract publihed in a High-performance regional conference, held in Maceió-  AL, Brazil, by "me" el. al. ( :) ) below (in Portuguese) .
 
Desempenho do Modelo de Ising Bidimensional em GPU e CPU
http://erad.lccv.ufal.br/?pg=anais
