#include <mpi.h>
#include <stdio.h>
#include <math.h>

#define _PRECISION_ 1000000000;

int main(int argc, char **argv)
{
	double precision = _PRECISION_;
	int  my_rank, proc_count;
	double pi_final;

	//My variables
	int myN, myStopN, rangeLength;
	double myLocalSum, nThElement;

	//Initialize MPI
	MPI_Init(&argc, &argv);

	//Get process id
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

	//Get number of all processes
	MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

	// Distribute required precision
	if(precision < proc_count)
	{
		printf("Precision smaller than the number of processes - try again");
		MPI_Finalize();
		return -1;
	}

	rangeLength = (int) floor(precision/proc_count);
	//Leibnitz formula instead of randomness
	myN = i * rangeLength;
	myStopN = (i +1) * rangeLength;

	// I am the last one, going to the end
	if(i == proc_count -1)
	{
		myStopN = (int) precision;
	}

	myLocalSum = 0;
	while(myN < myStopN)
	{
		nThElement = ( ( pow(-1.0, myN) ) / ( (2 * myN) + 1 ) );
		myLocalSum = myLocalSum + nThElement;
		myN = myN + proc_count;
	}

	// printf("\n Process rank %d  \n Process sign %d \n Process mine: %d \n", my_rank, sign, mine);
	// fflush(stdout);

	// Mergin all result to that in proc id = 0
	MPI_Reduce(&myLocalSum, &pi_final,1,MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);

	// In human speech - if proc id = 0
	if(!my_rank)
	{
		pi_final *= 4;
		printf("pi = %f", pi_final);
	}

	//Shutdown MPI
	MPI_Finalize();
	return 0;
}
