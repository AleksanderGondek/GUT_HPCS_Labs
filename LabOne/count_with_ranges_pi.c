#include <mpi.h>
#include <stdio.h>

#define _PRECISION_ 1000000000;

int main(int argc, char **argv)
{
	double precision = _PRECISION_;
	int  my_rank, proc_count;
	int mine, sign;
	double pi, pi_final;
	int i;

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

	//Each process do computing for itself
	pi = 0;
	mine = (precision/proccount)*myrank+1;
	sign = (( (mine-1)/2 ) % 2) ? -1 : 1;

	sign=(((mine-1)/2)%2)?-1:1;
	for (;mine<(precision/proccount)*(myrank+1);) 
	{
		// printf("\nProcess %d %d %d", myrank,sign,mine);
		// fflush(stdout);
		pi+=sign/(double)mine;
		mine+=+2;
		sign*=(-1);
	}


	// Mergin all result to that in proc id = 0

	MPI_Reduce(&pi, &pi_final,1,MPI_DOUBLE, MPI_SUM,0, MPI_COMM_WORLD);

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
