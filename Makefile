all:
	gcc -std=c11 -fopenmp -O3 driver.c winograd.c -o winograd
