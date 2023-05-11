#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void blurImage(Mat& image, int radius) {
    Mat blurredImage;
    blur(image, blurredImage, Size(radius, radius));
    blurredImage.copyTo(image);  // Copy the blurred image back to the original image
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank, numProcesses;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);

    // // Check if the number of processes is equal to the number of threads required
    // if (numProcesses != omp_get_num_procs()) {
    //     if (rank == 0) {
    //         printf("Error: Number of processes must be equal to the number of threads.\n");
    //     }
    //     MPI_Finalize();
    //     return EXIT_FAILURE;
    // }

    // Load image using OpenCV (performed by the master process)
    Mat image;
    if (rank == 0) {
        image = imread("lena.png", IMREAD_COLOR);
        if (image.empty()) {
            printf("Error: Failed to load image.\n");
            MPI_Finalize();
            return EXIT_FAILURE;
        }
    }

    // Broadcast image dimensions to all processes
    int imageWidth, imageHeight;
    if (rank == 0) {
        imageWidth = image.cols;
        imageHeight = image.rows;
    }
    MPI_Bcast(&imageWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&imageHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Divide the image into equal chunks among processes
    int chunkHeight = imageHeight / numProcesses;
    int startRow = rank * chunkHeight;
    int endRow = startRow + chunkHeight;

    // Allocate memory for local image chunk
    Mat localImage(chunkHeight, imageWidth, CV_8UC3);

    // Scatter the image data to all processes
    MPI_Scatter(image.data, chunkHeight * imageWidth * 3, MPI_BYTE,
                localImage.data, chunkHeight * imageWidth * 3, MPI_BYTE,
                0, MPI_COMM_WORLD);

    // Perform image blurring
    int blurRadius = 5;  // Adjust the blur radius as needed
    blurImage(localImage, blurRadius);

    // Gather the blurred image data from all processes
    MPI_Gather(localImage.data, chunkHeight * imageWidth * 3, MPI_BYTE,
               image.data, chunkHeight * imageWidth * 3, MPI_BYTE,
               0, MPI_COMM_WORLD);

    // Save the final blurred image (performed by the master process)
    if (rank == 0) {
        imwrite("blurred_image.png", image);
    }

    MPI_Finalize();

    return EXIT_SUCCESS;
}
