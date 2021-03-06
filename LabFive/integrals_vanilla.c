#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define _PRECISION_ 0.1;
#define _RANGE_SIZE_ 5

#define DATA 0
#define RESULT 1
#define FINISH 2

#define DEBUG

double function_to_integrate(double x)
{
    // Degrees to radians
    return sin(x * (M_PI / 180));
}

double integrate_range(double start, double end, double precision)
{
    double sum = 0;
    int i = 0;
    double a = _PRECISION_;

    for(i = start; i < end; i += a)
    {
        sum += function_to_integrate(i);
    }

    return sum;
}

int main(int argc, char **argv)
{
    MPI_Request *requests;
    int request_count=0;
    int request_completed;
    
    double precision = _PRECISION_;
    int  my_rank, proc_count;

    double start = 1;
    double end = 50;

    double *ranges;
    double range[2];

    double result = 0 ;
    double *result_temp;

    int sent_count = 0;
    int recv_count = 0;
    int i;
    
    MPI_Status status;
    
    //Initialize MPI
    MPI_Init(&argc, &argv);
    //Get process id
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    //Get number of all processes
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    if (proc_count<2) 
    {
        printf("Run with at least 2 processes");
        MPI_Finalize();
        return -1;
    }

    if ( ((end - start)/_RANGE_SIZE_) < 2*(proc_count-1)) 
    {
        printf("More subranges needed");
        MPI_Finalize();
        return -1;
    }

    // Master part of code
    if (my_rank==0) 
    {
        requests = (MPI_Request *) malloc(3*(proc_count-1)*sizeof(MPI_Request));        
        if (!requests) 
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }

        ranges=(double *)malloc(4*(proc_count-1)*sizeof(double));
        if (!ranges)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }
        
        result_temp=(double *)malloc((proc_count-1)*sizeof(double));
        if (!result_temp)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }

        // Send initial range to all slaves
        range[0] = start;
        for(i = 1; i < proc_count; i++)
        {
            range[1] = range[0] + _RANGE_SIZE_;

            #ifdef DEBUG
            printf("\nMaster sending range %f,%f to process %d",range[0], range[1], i);
            fflush(stdout);
            #endif
            
            MPI_Send(range, 2, MPI_DOUBLE, i, DATA, MPI_COMM_WORLD);
            
            sent_count++;
            range[0] = range[1];
        }

        // The first proc_count requests will be for receiving, the latter ones for sending
        for(i = 0; i < 2*(proc_count - 1); i++)
        {
            requests[i] = MPI_REQUEST_NULL;
        }

        // Start receiving for results from the slaves
        for(i = 1; i < proc_count; i++)
        {
            MPI_Irecv(&(result_temp[i-1]), 1, MPI_DOUBLE, i, RESULT, MPI_COMM_WORLD, &(requests[i-1]));
        }

        // Start sending new data parts to the slaves
        for(i = 1; i < proc_count; i++)
        {
            range[1] = range[0] + _RANGE_SIZE_;

            #ifdef DEBUG
            printf("\nMaster sending range %f,%f to process %d",range[0],range[1],i);
            fflush(stdout);
            #endif

            ranges[2*i-2] = range[0];
            ranges[2*i-1] = range[1];

            // Send it to process i
            MPI_Isend(&(ranges[2*i-2]), 2, MPI_DOUBLE, i, DATA,MPI_COMM_WORLD, &(requests[proc_count-2+i]));
            sent_count++;
            range[0] = range[1];
        }

        while(range[1] < end)
        {
            #ifdef DEBUG
            printf("\nMaster waiting for completion of requests");
            fflush(stdout);
            #endif

            // Wait for completion of any of the requests
            MPI_Waitany(2*proc_count-2, requests, &request_completed, MPI_STATUS_IGNORE);

            // If it is a result then send new data to the process
            // And add the result
            if (request_completed < (proc_count-1))
            {
                result += result_temp[request_completed];
                recv_count++;

                #ifdef DEBUG
                printf("\nMaster received %d result %f from process %d", recv_count, result_temp[request_completed], request_completed+1);
                fflush(stdout);
                #endif

                // First check if the send has terminated
                MPI_Wait(&(requests[proc_count-1 + request_completed]), MPI_STATUS_IGNORE);

                // Now send some new data portion to this process
                range[1] = range[0] + _RANGE_SIZE_;
                if (range[1]>end)
                {
                    range[1]=end;
                }

                #ifdef DEBUG
                printf("\nMaster sending range %f,%f to process %d", range[0], range[1], request_completed+1);
                fflush(stdout);
                #endif

                ranges[2*request_completed] = range[0];
                ranges[2*request_completed+1] = range[1];

                MPI_Isend(&(ranges[2*request_completed]), 2, MPI_DOUBLE, request_completed+1, DATA, MPI_COMM_WORLD, &(requests[proc_count-1+request_completed]));
                sent_count++;
                range[0]=range[1];

                // Now issue a corresponding recv
                MPI_Irecv(&(result_temp[request_completed]), 1, MPI_DOUBLE, request_completed+1, RESULT, MPI_COMM_WORLD, &(requests[request_completed]));
            }
        }

        // Now send the FINISHING ranges to the slaves
        // Shut down the slaves
        range[0] = range[1];
        for(i = 1; i < proc_count; i++)
        {
            #ifdef DEBUG
            printf("\nMaster sending FINISHING range %f,%f to process %d", range[0], range[1], i);
            fflush(stdout);
            #endif

            ranges[2*i-4+2*proc_count] = range[0];
            ranges[2*i-3+2*proc_count] = range[1];
            MPI_Isend(range, 2, MPI_DOUBLE, i, DATA, MPI_COMM_WORLD, &(requests[2*proc_count-3+i]));
        }

        // Now receive results from the processes - that is finalize the pending requests
        MPI_Waitall(3*proc_count-3, requests, MPI_STATUSES_IGNORE);

        // Now simply add the results
        for(i=0; i<(proc_count-1); i++)
        {
            result += result_temp[i];
        }

        #ifdef DEBUG
        printf("\nMaster before MPI_Waitall with total proccount=%d",proc_count);
        fflush(stdout);
        #endif

        // Now receive results for the initial sends
        for(i=0; i<(proc_count-1); i++)
        {
            #ifdef DEBUG
            printf("\nMaster receiving result from process %d",i+1);
            fflush(stdout);
            #endif
            
            MPI_Recv(&(result_temp[i]), 1, MPI_DOUBLE, i+1, RESULT, MPI_COMM_WORLD, &status);
            result += result_temp[i];
            recv_count++;

            #ifdef DEBUG
            printf("\nMaster received %d result %f from process %d", recv_count, result_temp[i], i+1);
            fflush(stdout);
            #endif
        }

        // Now display the result
        printf("\nHi, I am process 0, the result is %f\n", result);
    }
    else // Slave part of code
    {
        requests=(MPI_Request *)malloc(2*sizeof(MPI_Request));
        if (!requests)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }

        requests[0] = requests[1] = MPI_REQUEST_NULL;

        ranges=(double *)malloc(2*sizeof(double));
        if (!ranges)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }

        result_temp=(double *)malloc(2*sizeof(double));
        if (!result_temp)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }

        // First receive the initial data
        MPI_Recv(range, 2, MPI_DOUBLE, 0, DATA, MPI_COMM_WORLD, &status);

        #ifdef DEBUG
        printf("\nSlave received range %f,%f", range[0], range[1]);
        fflush(stdout);
        #endif

        // If there is any data to process
        while (range[0]<range[1])
        {
            // Before computing the next part start receiving a new data part
            MPI_Irecv(ranges, 2, MPI_DOUBLE, 0, DATA, MPI_COMM_WORLD, &(requests[0]));

            // Compute my part
            result_temp[1] = integrate_range(range[0], range[1], 1);

            #ifdef DEBUG
            printf("\nSlave %d just computed range %f,%f - result %f", my_rank, range[0], range[1], result_temp[1]);
            fflush(stdout);
            #endif

            // Now finish receiving the new part
            // And finish sending the previous results back to the master
            MPI_Waitall(2, requests, MPI_STATUSES_IGNORE);

            #ifdef DEBUG
            printf("\nSlave just received range %f,%f", ranges[0], ranges[1]);
            fflush(stdout);
            #endif

            range[0] = ranges[0];
            range[1] = ranges[1];
            result_temp[0] = result_temp[1];
            
            // And start sending the results back
            MPI_Isend(&result_temp[0], 1, MPI_DOUBLE, 0, RESULT, MPI_COMM_WORLD, &(requests[1]));   

            #ifdef DEBUG
            printf("\nSlave just initiated send to master with result %f", result_temp[0]);
            fflush(stdout);
            #endif         
        }

        // Now finish sending the last results to the master
        MPI_Wait(&(requests[1]), MPI_STATUS_IGNORE);
    }

    //Shutdown MPI
    MPI_Finalize();

    #ifdef DEBUG
    printf("\nProcess %d finished", my_rank);
    fflush(stdout);
    #endif

    return 0;
}
