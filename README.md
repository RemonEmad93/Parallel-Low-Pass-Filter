
(run on linux)

sequential code:

    command to compile:
        g++ -I/usr/local/include/opencv4 -o LPF_seq_RGB LPF_seq_RGB.cpp `pkg-config --cflags --libs opencv4`

    command to run:
        ./LPF_seq_RGB

    input: 
        image name
        kernel size

    ouput:
        original image
        filtered image
        duration of execution


openMP code:

    command to compile:
        g++ -I/usr/local/include/opencv4 -fopenmp -o LPF_openMP_RGB LPF_openMP_RGB.cpp `pkg-config --cflags --libs opencv4`

    command to run:
        ./LPF_openMP_RGB

    input: 
        image name
        kernel size
        number of threads
        
    ouput:
        original image
        filtered image
        duration of execution


MPI code:

    command to compile:
        mpic++ -I/usr/local/include/opencv4 -o LPF_MPI_RGB LPF_MPI_RGB.cpp `pkg-config --cflags --libs opencv4`

    command to run:
        mpirun -np no._of_threads ./LPF_MPI_RGB imageName kernelSize
        
    ouput:
        filtered image 
        duration of execution
