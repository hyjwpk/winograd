all:
	clang -std=c11 -fopenmp -Ofast -mcpu=native driver.c winograd.c -o winograd
# clang -std=c11 -fopenmp -Ofast -mcpu=native -I /shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/include -L /shareofs/apps/libs/kml/2.2.0-bisheng3.2.0/lib/kblas/nolocking -lkblas driver.c winograd.c -o winograd
# gcc -std=c11 -fopenmp -O3 driver.c winograd.c -o winograd