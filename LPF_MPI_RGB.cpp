// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string>
// #include <mpi.h>

// using namespace std;
// using namespace cv;

// int main() {
//     int rank, size;
//     int rows, cols;
//     int tag = 0;
//     Mat img, filtered_img;
//     string imagePath;
    

//     MPI_Init(NULL, NULL);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (rank == 0) {
//         cout << "Enter the file path of the image: ";
//         cin >> imagePath;
//         img = imread(imagePath, IMREAD_UNCHANGED);
//         if (img.empty()) {
//             cerr << "Error: Could not read image file." << endl;
//             return 1;
//         }
//         rows = img.rows;
//         cols = img.cols;
//     }

//     MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     int strip_height = rows / size;
//     int strip_size = strip_height * cols * img.channels();
//     uchar* strip_data = new uchar[strip_size];

//     MPI_Scatter(img.data, strip_size, MPI_UNSIGNED_CHAR,
//                 strip_data, strip_size, MPI_UNSIGNED_CHAR,
//                 0, MPI_COMM_WORLD);

//     Mat strip(strip_height, cols, img.type(), strip_data);

//     Mat filtered_strip;
//     bilateralFilter(strip, filtered_strip, 9, 75, 75);

//     uchar* filtered_strip_data = filtered_strip.data;

//     uchar* recv_data = NULL;
//     if (rank == 0) {
//         recv_data = new uchar[rows * cols * img.channels()];
//     }

//     MPI_Gather(filtered_strip_data, strip_size, MPI_UNSIGNED_CHAR,
//                recv_data, strip_size, MPI_UNSIGNED_CHAR,
//                0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         filtered_img = Mat(rows, cols, img.type(), recv_data);
//         delete[] recv_data;
//     }

//     delete[] strip_data;

//     MPI_Finalize();

//     if (rank == 0) {
//         namedWindow("Original image", WINDOW_NORMAL);
//         namedWindow("Filtered image", WINDOW_NORMAL);
//         imshow("Original image", img);
//         imshow("Filtered image", filtered_img);
//         waitKey(0);
//     }

//     return 0;
// }
//////////////////////////////////////////////////////////////////////////////
// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <mpi.h>

// using namespace cv;
// using namespace std;

// int main(int argc, char** argv)
// {
//     int rank, size;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     Mat image;

//     if (rank == 0) {
//         string img_path;
//         cout << "Enter the image path: ";
//         cin >> img_path;
//         image = imread(img_path);
//     }

//     int rows = image.rows;
//     int cols = image.cols;
//     int channels = image.channels();
//     int rows_per_process = rows / size;

//     MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&rows_per_process, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     if (rank != 0) {
//         image.create(rows, cols, CV_8UC3);
//     }

//     unsigned char* send_buf = new unsigned char[rows_per_process * cols * channels];
//     MPI_Scatter(image.data, rows_per_process * cols * channels, MPI_UNSIGNED_CHAR,
//         send_buf, rows_per_process * cols * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

//     unsigned char* recv_buf = new unsigned char[rows_per_process * cols * channels];
//     for (int i = 0; i < rows_per_process; i++) {
//         int offset = i * cols * channels;
//         for (int j = 0; j < cols; j++) {
//             int k = offset + j * channels;
//             for (int c = 0; c < channels; c++) {
//                 if (i == 0) {
//                     recv_buf[k + c] = send_buf[k + c];
//                 } else {
//                     recv_buf[k + c] = (unsigned char)(0.2 * send_buf[k + c] + 0.6 * recv_buf[k - cols * channels + c] + 0.2 * recv_buf[k + c - channels]);
//                 }
//             }
//         }
//     }

//     MPI_Gather(recv_buf, rows_per_process * cols * channels, MPI_UNSIGNED_CHAR,
//         image.data, rows_per_process * cols * channels, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

//     delete[] send_buf;
//     delete[] recv_buf;

//     if (rank == 0) {
//         imshow("Original Image", image);
//         waitKey(0);
//         destroyAllWindows();
//     }

//     MPI_Finalize();

//     return 0;
// }

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string>
// #include <chrono>
// #include <mpi.h>

// using namespace std;
// using namespace cv;
// using namespace chrono;


// int main(int argc, char** argv) {

//     Mat *image;
//     // Initialize MPI
//     MPI_Init(&argc, &argv);

//     // Get the number of processes
//     int num_processes;
//     MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

//     // Get the rank of the current process
//     int rank ;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);


//     // Check if the current process is the root process
//     if (rank == 0) {

//         // Ask user for image path
//         string imagePath;
//         cout << "Enter the image name: ";
//         cin >> imagePath;

//         // Load the image
//         image = imread(imagePath);

//         // check the entered image path
//         if (image.empty()) {
//             cerr << "Error: Could not read image file." << endl;
//             return 1;
//         }

//         // Ask user for kernel size
//         int Ksize;
//         cout << "Enter the kernel size: ";
//         cin >> Ksize;

//         // Check that the kernel number entered is odd
//         while(true)
//         {
//             if (Ksize % 2 == 1) {
//                 break;
//             }else{
//                 cout << "the kernel size must be odd number, Enter the kernel size: ";
//                 cin >> Ksize;
//             }
//         }

//         // Create a kernel for blurring
//         int k = Ksize / 2;
//         Mat kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);
    
//         // Create a copy of the image to hold the blurred image
//         Mat blurred_image = image.clone();
    
//         // start timer
//         auto start_time = high_resolution_clock::now();

//         // Divide the image into blocks
//         int block_size = image.cols / num_processes;
//         int start_x = rank * block_size;
//         int end_x = min(image.cols, (rank + 1) * block_size);

//         // Apply the kernel to each block of the image
//         for (int i = start_x; i < end_x; i++) {
//             for (int j = k; j < image.rows - k; j++) {
//                 float sum_r = 0;
//                 float sum_g = 0;
//                 float sum_b = 0;
//                 for (int x = -k; x <= k; x++) {
//                     for (int y = -k; y <= k; y++) {
//                         Vec3b pixel = image.at<Vec3b>(i + x, j + y);
//                         sum_b += pixel[0] * kernel.at<float>(k + x, k + y);
//                         sum_g += pixel[1] * kernel.at<float>(k + x, k + y);
//                         sum_r += pixel[2] * kernel.at<float>(k + x, k + y);
//                     }
//                 }
//                 Vec3b new_pixel(sum_b, sum_g, sum_r);
//                 blurred_image.at<Vec3b>(i, j) = new_pixel;
//             }
//         }

//         // end timer
//         auto end_time = high_resolution_clock::now();

//         // show both clear and blurred image
//         namedWindow("Original image", WINDOW_NORMAL);
//         namedWindow("Filtered image", WINDOW_NORMAL);
//         imshow("Original image", image);
//         imshow("Filtered image", blurred_image);

//         // show the execution time in seconds
//         cout << "Execution time: " << duration_cast<microseconds>(end_time - start_time).count() / 1000000.0 << " seconds\n";

//         waitKey(0);

//     } else {


//         // Wait for the root process to send the image
//         MPI_Recv(&image, sizeof(image), MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//         // Apply the kernel to the image
//         for (int i = start_x; i < end_x; i++) {
//             for (int j = k; j < image.rows - k; j++) {
//                 float sum_r = 0;
//                 float sum_g = 0;
//                 float sum_b = 0;
//                 for (int x = -k; x <= k; x++) {
//                     for (int y = -k; y <= k; y++) {
//                         Vec3b pixel = image.at<Vec3b>(i + x, j + y);
//                         sum_b += pixel[0] * kernel.at<float>(k + x, k + y);
//                         sum_g += pixel[1] * kernel.at<float>(k + x, k + y);
//                         sum_r += pixel[2] * kernel.at<float>(k + x, k + y);
//                     }
//                 }
//                 Vec3b new_pixel(sum_b, sum_g, sum_r);
//                 blurred_image.at<Vec3b>(i, j) = new_pixel;
//             }
//         }

//         // Send the blurred image back to the root process
//         MPI_Send(&blurred_image, sizeof(blurred_image), MPI_CHAR, 0, 0, MPI_COMM_WORLD);

//     }

//     // Finalize MPI
//     MPI_Finalize();

//     return 0;
// }


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string>
// #include <chrono>
// #include <mpi.h>

// using namespace std;
// using namespace cv;
// using namespace chrono;

// // Function to blur the image
// void blurImage(Mat& image, Mat& blurred_image, int Ksize, int rank, int size) {

//     // Create a kernel for blurring
//     int k = Ksize / 2;
//     Mat kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);

//     // Determine the start and end rows to process for this process
//     int start_row = rank * (image.rows / size);
//     int end_row = (rank== size-1)? image.rows: (rank + 1) * (image.rows / size);

//     // Apply the kernel to each pixel in the image
//     for (int i = start_row + k; i < end_row - k; i++) {
//         for (int j = k; j < image.cols - k; j++) {
//             float sum_r = 0;
//             float sum_g = 0;
//             float sum_b = 0;
//             for (int x = -k; x <= k; x++) {
//                 for (int y = -k; y <= k; y++) {
//                     Vec3b pixel = image.at<Vec3b>(i + x, j + y);
//                     sum_b += pixel[0] * kernel.at<float>(k + x, k + y);
//                     sum_g += pixel[1] * kernel.at<float>(k + x, k + y);
//                     sum_r += pixel[2] * kernel.at<float>(k + x, k + y);
//                 }
//             }
//             Vec3b new_pixel(sum_b, sum_g, sum_r);
//             blurred_image.at<Vec3b>(i, j) = new_pixel;
//         }
//     }

//     // Synchronize all processes to make sure all have completed their work
//     MPI_Barrier(MPI_COMM_WORLD);

//     // If this is the first process, copy the edges of the blurred image to the next process
//     if (rank == 0) {
//         for (int i = 0; i < k; i++) {
//             MPI_Send(blurred_image.row(i + k).data, blurred_image.step, MPI_BYTE, rank + 1, 0, MPI_COMM_WORLD);
//         }
//     }

//     // If this is the last process, copy the edges of the blurred image to the previous process
//     else if (rank == size - 1) {
//         for (int i = 0; i < k; i++) {
//             MPI_Recv(blurred_image.row(i + start_row).data, blurred_image.step, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         }
//     }

//     // If this is a middle process, copy the edges of the blurred image to both the previous and next processes
//     else {
//         for (int i = 0; i < k; i++) {
//             MPI_Sendrecv(blurred_image.row(i + k).data, blurred_image.step, MPI_BYTE, rank + 1, 0,
//                 blurred_image.row(i + start_row - k).data, blurred_image.step, MPI_BYTE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         }
//     }
// }

// int main(int argc, char** argv) {

//     // Initialize MPI
//     MPI_Init(&argc, &argv);

//     // Get the rank and size of the current process
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     // Read in the image file
//     Mat image;
//     if (rank == 0) {

//         // ask user for image path
//         string imagePath;
//         cout << "Enter the image name: ";
//         cin >> imagePath;

//         image = imread(imagePath);
//         if (image.empty()) {
//             cerr << "Error: Could not open or find the image file" << endl;
//             MPI_Finalize();
//             return -1;
//         }
//     }

//     // Broadcast the image size to all processes
//     int rows, cols;
//     if (rank == 0) {
//         rows = image.rows;
//         cols = image.cols;
//     }
//     MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Create a matrix to hold the blurred image
//     Mat blurred_image(rows, cols, CV_8UC3);

//     // Broadcast the kernel size to all processes
//     int kernel_size = 3;
//     MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Blur the image
//     auto start = high_resolution_clock::now();
//     blurImage(image, blurred_image, kernel_size, rank, size);
//     auto stop = high_resolution_clock::now();

//     // Print the time taken for blurring the image
//     if (rank == 0) {
//         auto duration = duration_cast<milliseconds>(stop - start);
//         cout << "Time taken for blurring the image: " << duration.count() << " milliseconds" << endl;

//         // Display the original and blurred images side by side
//         Mat display_image;
//         hconcat(image, blurred_image, display_image);
//         namedWindow("Original Image - Blurred Image", WINDOW_NORMAL);
//         imshow("Original Image - Blurred Image", display_image);
//         waitKey(0);
//     }

//     // Finalize MPI
//     MPI_Finalize();

//     return 0;
// }

///////////////////////////////////////////////////////////////////////////////

// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string>
// #include <chrono>
// #include <mpi.h>

// using namespace std;
// using namespace cv;
// using namespace chrono;

// int main()
// {
//     MPI_Init(NULL, NULL);

//     int size, rank;
//     MPI_Comm_size(MPI_COMM_WORLD, &size);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     Mat image, kernel;
//     if (rank == 0) 
//     {
//         // ask user for image path
//         string imagePath;
//         cout << "Enter the image name: ";
//         cin >> imagePath;

//         image = imread(imagePath);
//         // check the entered image path
//         if (image.empty()) {
//             cerr << "Error: Could not read image file." << endl;
//             return 1;
//         }

//         // ask user for kernel size 
//         int Ksize;
//         cout << "Enter the kernel size: ";
//         cin >> Ksize;

//         // check that the kernel number entered is odd
//         while(true)
//         {
//             if (Ksize % 2 == 1) {
//                 break;
//             }else{
//                 cout << "the kernel size must be odd number, Enter the kernel size: ";
//                 cin >> Ksize;
//             }
//         }

//         // Create a kernel for blurring
//         int k = Ksize / 2;
//         kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);
//     }

//     // Broadcast the value of n to all processes
//     MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
// }

///////////////////////////////////////////////////////////////////


#include <iostream>
#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) {
            cout << "Usage: " << argv[0] << " <input_image_path> <kernel_size>" << endl;
        }
        MPI_Finalize();
        return 0;
    }

    if (rank == 0) {
        cout << "Starting image blurring..." << endl;
    }

    // Load input image
    Mat input_image = imread(argv[1], IMREAD_COLOR);

    if (input_image.empty()) {
        if (rank == 0) {
            cout << "Could not open or find the image: " << argv[1] << endl;
        }
        MPI_Finalize();
        return 0;
    }

    // Convert input image to grayscale
    Mat grayscale_image;
    cvtColor(input_image, grayscale_image, COLOR_BGR2GRAY);

    // Set kernel size for blurring
    int kernel_size = atoi(argv[2]);

    // Blur image using user-defined kernel size
    Mat blurred_image;
    blur(grayscale_image, blurred_image, Size(kernel_size, kernel_size));

    // Save output image
    imwrite("blurred_image.jpg", blurred_image);

    if (rank == 0) {
        cout << "Blurred image saved as blurred_image.jpg" << endl;
    }

    MPI_Finalize();

    return 0;
}
