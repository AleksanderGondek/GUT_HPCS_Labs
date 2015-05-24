#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define _EPSILON_ 0.0001
#define _INITIAL_PRECISION_ 0.01
#define _PRECISION_ICREASE_ 0.1

#define MSG_TAG_DATA 0
#define MSG_RETURN_DATA 1

#define SLAVE_STATUS_WAITING 0
#define SLAVE_STATUS_WORKING 1
#define SLAVE_STATUS_FINISHED 2

#define DEBUG

double function_to_integrate(double x)
{
    // Degrees to radians
    return sin(x * (M_PI / 180));
    //return 2.0f;
}

double integrate_range(double start, double end, double precision)
{
    double sum = 0;
    double i = 0;

    for(i = start; i < end; i += precision)
    {
        sum += function_to_integrate(i) * precision;
    }

    return sum;
}

// We are assuming that range is larger than number of processes
int main(int argc, char **argv)
{   
    int my_rank, proc_count, i, request_completed;
    
    double range_start = 0.0f;
    double range_end = 100.0f;
    double my_range_start;
    double my_range_end;
    
    double message[3];

    MPI_Status status;
    MPI_Request *requests;

    //Initialize MPI
    MPI_Init(&argc, &argv);
    //Get process id
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    //Get number of all processes
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    // Check initial constraints
    if (proc_count < 2) 
    {
        printf("\nRun with at least 2 processes");
        MPI_Finalize();
        return -1;
    }

    if ((range_end - range_start) < (proc_count - 1))
    {
        printf("\nYou need to set bigger range to calculate");
        MPI_Finalize();
        return -1;
    }

    // Calculate initial divide into ranges
    double range_jump = ((range_end - range_start)/(proc_count - 1));
    double statues[proc_count - 1][4];

    // If I am the master
    if(my_rank == 0)
    {
        // Initialise statuses array
        for(i = 0; i < (proc_count - 1); i++)
        {
            statues[i][0] = NAN; // Previous result
            statues[i][1] = NAN; // Latest result
            statues[i][2] = _INITIAL_PRECISION_; //Latest used precision
            statues[i][3] =  SLAVE_STATUS_WAITING; // Current status
        }

        // Initialize "routing" data
        requests = (MPI_Request *) malloc(2*(proc_count-1)*sizeof(MPI_Request));        
        if (!requests) 
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }

        // The first proc_count requests will be for receiving, the latter ones for sending
        for(i = 0; i < 2*(proc_count - 1); i++)
        {
            requests[i] = MPI_REQUEST_NULL;
        }

        // Sent ranges for each slave, with intial precision
        for(i= 1; i < proc_count; i++)
        {
            // Get ranges for slave
            message[0] = range_start + ((i-1) * range_jump);
            message[1] = message[0] + range_jump;
            message[2] = statues[i-1][2];

            #ifdef DEBUG
            printf("\nMASTER sending range %f,%f with precision %f to process %d", message[0], message[1], message[2], i);
            fflush(stdout);
            #endif

            MPI_Send(message, 3, MPI_DOUBLE, i, MSG_TAG_DATA, MPI_COMM_WORLD);
            statues[i-1][3] = SLAVE_STATUS_WORKING;
        }

        // Start receiving for results from the slaves
        for(i = 1; i < proc_count; i++)
        {
            MPI_Irecv(&(statues[i-1][1]), 1, MPI_DOUBLE, i, MSG_RETURN_DATA, MPI_COMM_WORLD, &(requests[i-1]));
        }

        int should_continue = proc_count - 1;
        while(should_continue > 0)
        {
            #ifdef DEBUG
            printf("\nMASTER waiting for completion of requests");
            fflush(stdout);
            #endif

            // Wait for completion of any of the requests
            MPI_Waitany(2*proc_count-2, requests, &request_completed, MPI_STATUS_IGNORE);

            #ifdef DEBUG
            printf("\nMASTER passed wait any");
            fflush(stdout);
            #endif

            // If it is a result then send new data to the process
            // And add the result
            if (request_completed < (proc_count-1))
            {
                #ifdef DEBUG
                printf("\nMASTER received result %f from process %d", statues[request_completed][1], request_completed+1);
                fflush(stdout);
                #endif

                double difference_for_eps = fabs(statues[request_completed][1]-statues[request_completed][0]);
                int check_eps = 1;
                if(isnan(statues[request_completed][1]) || isnan(statues[request_completed][0]))
                {
                    check_eps = 0;
                }
                if(statues[request_completed][1] == statues[request_completed][0])
                {
                    check_eps = 0;
                }
                
                #ifdef DEBUG
                printf("\nMASTER check_eps %d, difference %f", check_eps, difference_for_eps);
                #endif
                
                if(check_eps && (difference_for_eps < _EPSILON_))
                {
                    //If Stop condition
                    #ifdef DEBUG
                    printf("\nMASTER sending TERMINATION SIGNAL to process %d", request_completed+1);
                    fflush(stdout);
                    #endif

                    // First check if the send has terminated
                    // Because PROC_COUNT-1 is a shift right, 'new table', ffs
                    MPI_Wait(&(requests[proc_count-1 + request_completed]), MPI_STATUS_IGNORE);

                    // Get data for slave
                    message[0] = NAN;
                    message[1] = NAN;
                    message[2] = NAN;

                    // Because PROC_COUNT-1 is a shift right, 'new table', ffs, this separates receive request form send requests
                    MPI_Isend(&message, 3, MPI_DOUBLE, request_completed+1, MSG_TAG_DATA, MPI_COMM_WORLD, &(requests[proc_count-1+request_completed]));
                    // We don't expect response
                    requests[request_completed] = MPI_REQUEST_NULL;

                    // Set Slave status to finished
                    statues[request_completed][3] = SLAVE_STATUS_FINISHED;
                }
                else
                {
                    // First check if the send has terminated
                    // Because PROC_COUNT-1 is a shift right, 'new table', ffs
                    MPI_Wait(&(requests[proc_count-1 + request_completed]), MPI_STATUS_IGNORE);

                    // Get data for slave
                    message[0] = NAN;
                    message[1] = NAN;
                    statues[request_completed][2] = statues[request_completed][2] * _PRECISION_ICREASE_;
                    message[2] = statues[request_completed][2];

                    #ifdef DEBUG
                    printf("\nMASTER sending range %f,%f, with precision %f to process %d", message[0], message[1], message[2], request_completed+1);
                    fflush(stdout);
                    #endif

                    // Because PROC_COUNT-1 is a shift right, 'new table', ffs, this separates receive request form send requests
                    MPI_Isend(&message, 3, MPI_DOUBLE, request_completed+1, MSG_TAG_DATA, MPI_COMM_WORLD, &(requests[proc_count-1+request_completed]));
                    statues[request_completed][3] = SLAVE_STATUS_WORKING;

                    // Now issue a corresponding recv
                    MPI_Irecv(&(statues[request_completed][1]), 1, MPI_DOUBLE, request_completed+1, MSG_RETURN_DATA, MPI_COMM_WORLD, &(requests[request_completed]));
                }

                statues[request_completed][0] = statues[request_completed][1];
            }

            // Check if everyone has finished working
            for(i = 0; i < proc_count - 1; i++)
            {
                #ifdef DEBUG
                printf("\nMASTER checking if statues[%d][3] equals 2, meaning slave has finished working - current value: %f", i, statues[i][3]);
                #endif
                if(statues[i][3] == SLAVE_STATUS_FINISHED)
                {
                    should_continue = should_continue - 1;
                }
            }
            if(should_continue > 0)
            {
                should_continue = proc_count - 1;
                #ifdef DEBUG
                printf("\nMASTER reseting should_continue value");
                #endif
            }
        }

        for(i = 0; i < proc_count - 1; i++)
        {
            #ifdef DEBUG
            printf("\nHi, I am process 0, statues[%d][0] is %f\n", i, statues[i][0]);
            printf("\nHi, I am process 0, statues[%d][1] is %f\n", i, statues[i][1]);
            printf("\nHi, I am process 0, statues[%d][2] is %f\n", i, statues[i][2]);
            printf("\nHi, I am process 0, statues[%d][3] is %f\n", i, statues[i][3]);
            #endif
        }

        //Calculate the final result
        double final_result = 0.0f;
        for(i = 0; i < proc_count - 1; i++)
        {
            //Add latest returned value
            final_result += statues[i][1];
        }

        printf("\n MASTER Result: %f", final_result);
    }
    //I am the slave
    else
    {
        int firstRun = 1;

        // Initialise requests
        requests=(MPI_Request *)malloc(2*sizeof(MPI_Request));
        if (!requests)
        {
            printf("\nNot enough memory");
            MPI_Finalize();
            return -1;
        }
        //Default value
        requests[0] = requests[1] = MPI_REQUEST_NULL;

        //Receive given range for process
        MPI_Recv(message, 3, MPI_DOUBLE, 0, MSG_TAG_DATA, MPI_COMM_WORLD, &status);

        #ifdef DEBUG
        printf("\nSLAVE NO. %d received range %f,%f with precision %f", my_rank, message[0], message[1], message[2]);
        fflush(stdout);
        #endif

        my_range_start = message[0];
        my_range_end = message[1];

        int should_work = 1;
        while(should_work == 1)
        {
            if(!firstRun)
            {
                // Before computing the next part start receiving a new data part
                MPI_Irecv(message, 3, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &(requests[0]));
            }

            // Compute integral
            double computation_results = integrate_range(my_range_start, my_range_end, message[2]);

            #ifdef DEBUG
            printf("\nSLAVE NO. %d just computed range %f,%f - result %f", my_rank, my_range_start, my_range_end, computation_results);
            fflush(stdout);
            #endif

            if(!firstRun)
            {
                // Now finish receiving the new part
                // And finish sending the previous results back to the master
                MPI_Waitall(2, requests, MPI_STATUSES_IGNORE); 
            }

            #ifdef DEBUG
            printf("\nSLAVE NO. %d received range %f,%f with precision %f", my_rank, message[0], message[1], message[2]);
            fflush(stdout);
            #endif

            if(isnan(message[2]))
            {
                #ifdef DEBUG
                printf("\nSLAVE NO. %d received TERMINATION SIGNAL", my_rank);
                fflush(stdout);
                #endif
                should_work = 0;
            }
            
            // And start sending the results back
            MPI_Isend(&computation_results, 1, MPI_DOUBLE, 0, MSG_RETURN_DATA, MPI_COMM_WORLD, &(requests[1]));

            #ifdef DEBUG
            printf("\nSLAVE NO. %d just initiated sending to master with result %f", my_rank, computation_results);
            fflush(stdout);
            #endif 

            firstRun = 0;
        }

        // Now finish sending the last results to the master
        MPI_Wait(&(requests[1]), MPI_STATUS_IGNORE);
    }

    //double zenon = integrate_range(0.0f, 5.0f, 0.0001f);

    //Shutdown MPI
    MPI_Finalize();

    return 0;
}
