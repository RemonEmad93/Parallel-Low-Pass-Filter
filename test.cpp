#include <iostream>
#include <mpi.h>
#include <cstdlib>
#include <ctime>

using namespace std;

int main()
{
    MPI_Init(NULL, NULL);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n; // size of vector and matrix
    if (rank == 0) 
    {
        cout << "Enter the value of n: ";
        cin >> n;
    }

    // Broadcast the value of n to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Define the vector and the matrix
    int vector[n], matrix[n][n];
    // fill them with random numbers between 1 and 10
    srand(time(NULL)); 
    for (int i = 0; i < n; i++) 
    {
        vector[i] = rand() % 10 + 1; 
        for (int j = 0; j < n; j++)
            matrix[i][j] = rand() % 10 + 1; 
    }

    // Define the result vector
    double result[n] = {0};

    // Each process calculates its portion
    int portion= n / size;
    int start= rank * portion;
    int end = start + portion;
    if (rank == size - 1)
        // Last process takes care of the remaining cols
        end = n;

    //calculate the local result for each process
    for (int i = start; i < end; i++) 
        for (int j = 0; j < n; j++)
            result[i] += matrix[j][i] * vector[j];
    

    // Gather the results from each process
    int counts[size];
    int start_index[size];
    for (int i = 0; i < size; i++) 
    {
        if (i == size - 1)
            counts[i] = n - (portion * (size - 1));
        else
            counts[i] = portion;

        start_index[i] = i * portion;
    }
    MPI_Gatherv(&result[start], end - start, MPI_DOUBLE, &result[0], counts, start_index, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // print the vector, matrix and result on process 0
    if (rank == 0) 
    {
        cout << "the vector: "<< endl;
        for(int i=0; i<n; i++)
            cout << vector[i]<< " ";
        cout <<endl;

        cout <<"the matrix: "<< endl;
        for (int i = 0; i < n; i++) 
        {
            for(int j=0; j<n; j++)
                cout << matrix[i][j]<< " ";
            
            cout << endl;
        }  

        cout << "the result: "<< endl;
        for (int i = 0; i < n; i++)
            cout << result[i] << " ";

        cout << endl;
    }

    MPI_Finalize();
    return 0;
}