
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <chrono>
#include <mpi.h>

using namespace std;
using namespace cv;
using namespace chrono;

int main(int argc, char* argv[]) {


    // cout << "Enter the image name:1 ";
    // int input = cin.get();

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size of the MPI processes
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    Mat image = imread("lena.png");

    // Ask user for image path (only rank 0 does this)
    // string imagePath;
    // if (rank == 0) {
    //     cout << "Enter the image name: ";
    //     cin >> imagePath;
    // }

    // Broadcast the size of imagePath to all processes
    // int imagePathSize = 0;
    // if (rank == 0) {
    //     imagePathSize = imagePath.size();
    // }
    // MPI_Bcast(&imagePathSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Resize the imagePath string on each process to accommodate the received size
    // imagePath.resize(imagePathSize);

    // // Broadcast the imagePath string to all processes
    // MPI_Bcast(&imagePath[0], imagePathSize, MPI_CHAR, 0, MPI_COMM_WORLD);

    // Load the image (only rank 0 does this)
    // Mat image;
    // if (rank == 0) {
    //     image = imread(imagePath);

    //     // Check the entered image path
    //     if (image.empty()) {
    //         cerr << "Error: Could not read image file." << endl;
    //         MPI_Finalize();
    //         return 1;
    //     }
    // }

    // Broadcast the image dimensions to all processes
    // int imageRows, imageCols;
    // if (rank == 0) {
    //     imageRows = image.rows;
    //     imageCols = image.cols;
    // }
    // MPI_Bcast(&imageRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&imageCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Ask user for kernel size (only rank 0 does this)
    int Ksize;
    if (rank == 0) {
        cout << "Enter the kernel size: ";
        cin >> Ksize;

        // Check that the kernel number entered is odd
        while (Ksize < 3 || Ksize % 2 != 1) {
            cout << "The kernel size must be an odd number greater than or equal to 3. Enter the kernel size: ";
            cin >> Ksize;
        }
    }

    // Broadcast the kernel size to all processes
    MPI_Bcast(&Ksize, 1, MPI_INT, 0, MPI_COMM_WORLD);


    // Create a kernel for blurring (only rank 0 does this)
    Mat kernel;
    // if (rank == 0) {
    //     int k = Ksize / 2;
    //     kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);
    // }
        kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);

    // Calculate the chunk size for each process
    int chunkSize = (image.rows - Ksize + 1) / size;
    int remainder = (image.rows - Ksize + 1) % size;
    int startRow = rank * chunkSize;
    int endRow = startRow + chunkSize - 1;

    // Adjust the endRow for the last process
    if (rank == size - 1) {
        endRow += remainder;
    }

    // Create a copy of the image chunk to hold the blurred image chunk
    Mat blurred_image = image(Rect(0, startRow, image.cols, endRow - startRow + 1)).clone();

    // Start the timer (only rank 0 does this)
    double start_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }

    cout<< "herrere"<< endl;
    // Apply the kernel to each pixel in the image chunk
    for (int i = 0; i < blurred_image.rows; i++) {
        for (int j = Ksize / 2; j < blurred_image.cols - Ksize / 2; j++) {
            float sum_r = 0;
            float sum_g = 0;
            float sum_b = 0;
            for (int x = -Ksize / 2; x <= Ksize / 2; x++) {
                for (int y = -Ksize / 2; y <= Ksize / 2; y++) {
                    Vec3b pixel = image.at<Vec3b>(startRow + i + x, j + y);
                    sum_b += pixel[0] * kernel.at<float>(Ksize / 2 + x, Ksize / 2 + y);
                    sum_g += pixel[1] * kernel.at<float>(Ksize / 2 + x, Ksize / 2 + y);
                    sum_r += pixel[2] * kernel.at<float>(Ksize / 2 + x, Ksize / 2 + y);
                }
            }
            Vec3b new_pixel(sum_b, sum_g, sum_r);
            blurred_image.at<Vec3b>(i, j) = new_pixel;
        }
    }
    cout<< "herrere2"<< endl;

    // Gather the blurred image chunks from all processes to reconstruct the complete blurred image (only rank 0 does this)
    if (rank == 0) {
        vector<Mat> blurred_chunks(size);
        blurred_chunks[0] = blurred_image;
        for (int i = 1; i < size; i++) {
            blurred_chunks[i] = Mat(chunkSize, image.cols, CV_8UC3);
            MPI_Recv(blurred_chunks[i].data, blurred_chunks[i].total() * blurred_chunks[i].elemSize(), MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        // Concatenate the blurred image chunks
        vconcat(blurred_chunks, blurred_image);
    }
    else {
        // Send the blurred image chunk to rank 0
        MPI_Send(blurred_image.data, blurred_image.total() * blurred_image.elemSize(), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
    }

    // End the timer (only rank 0 does this)
    double end_time;
    if (rank == 0) {
        end_time = MPI_Wtime();
    }

    // Rank 0 displays the original and blurred images and prints the execution time
    if (rank == 0) {
        // Display the images
        namedWindow("Original image", WINDOW_NORMAL);
        namedWindow("Filtered image", WINDOW_NORMAL);
        imshow("Original image", image);
        imshow("Filtered image", blurred_image);

        // Show the execution time in seconds
        cout << "Execution time: " << end_time - start_time << " seconds\n";
    }

    // Wait for key press (all processes)
    waitKey(0);

    // Finalize MPI
    MPI_Finalize();

    return 0;
}
