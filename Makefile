all:
	clang -std=c11 -fopenmp -Ofast -mcpu=native driver.c winograd.c -o winograd
# gcc -std=c11 -fopenmp -O3 driver.c winograd.c -o winograd