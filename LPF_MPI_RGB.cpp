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

//     Mat image, kernel, blurred_image;
//     int k;
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
//         k = Ksize / 2;
//         kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);

//         blurred_image = image.clone();
//     }

//     // Broadcast the value of n to all processes
//     MPI_Bcast(&image, 1, MPI_INT, 0, MPI_COMM_WORLD);////////????????????
//     MPI_Bcast(&kernel, 1, MPI_INT, 0, MPI_COMM_WORLD);/////////??????????????????
//     MPI_Bcast(&blurred_image, 1, MPI_INT, 0, MPI_COMM_WORLD);///////////???????????
//     MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     int portion= k / size;
//     int start= rank * portion;
//     int end = start + portion;
//     if (rank == size - 1)
//         // Last process takes care of the remaining cols
//         end = n;


// }

///////////////////////////////////////////////////////////////////


// #include <iostream>
// #include <mpi.h>
// #include <opencv2/opencv.hpp>

// using namespace std;
// using namespace cv;

// int main(int argc, char** argv) {

//     MPI_Init(&argc, &argv);

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (argc < 3) {
//         if (rank == 0) {
//             cout << "Usage: " << argv[0] << " <input_image_path> <kernel_size>" << endl;
//         }
//         MPI_Finalize();
//         return 0;
//     }

//     if (rank == 0) {
//         cout << "Starting image blurring..." << endl;
//     }

//     // Load input image
//     Mat input_image = imread(argv[1], IMREAD_COLOR);

//     if (input_image.empty()) {
//         if (rank == 0) {
//             cout << "Could not open or find the image: " << argv[1] << endl;
//         }
//         MPI_Finalize();
//         return 0;
//     }

//     // Convert input image to grayscale
//     Mat grayscale_image;
//     cvtColor(input_image, grayscale_image, COLOR_BGR2GRAY);

//     // Set kernel size for blurring
//     int kernel_size = atoi(argv[2]);

//     // Blur image using user-defined kernel size
//     Mat blurred_image;
//     blur(grayscale_image, blurred_image, Size(kernel_size, kernel_size));

//     // Save output image
//     imwrite("blurred.png", blurred_image);

//     if (rank == 0) {
//         cout << "Blurred image saved as blurred_image.jpg" << endl;
//     }

//     MPI_Finalize();

//     return 0;
// }

///////////////////////////////////////////////////////////////////////////

// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <mpi.h>

// using namespace cv;

// int main(int argc, char **argv)
// {
//     int rank, size;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (rank == 0) {
//         // Ask user for input
//         std::string img_path;
//         std::cout << "Enter path to image: ";
//         std::cin >> img_path;
//         int kernel_size;
//         std::cout << "Enter kernel size (odd number): ";
//         std::cin >> kernel_size;

//         // Load image
//         Mat image = imread(img_path, IMREAD_COLOR);
//         if (image.empty()) {
//             std::cerr << "Error: Failed to load image.\n";
//             MPI_Abort(MPI_COMM_WORLD, 1);
//             return 1;
//         }

//         // Distribute image to other processes
//         int rows_per_process = image.rows / size;
//         for (int i = 1; i < size; i++) {
//             int offset = i * rows_per_process;
//             int rows = (i == size - 1) ? image.rows - offset : rows_per_process;
//             MPI_Send(&rows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
//             MPI_Send(image.ptr(offset), rows * image.step, MPI_BYTE, i, 0, MPI_COMM_WORLD);
//         }

//         // Process part of image on root process
//         int offset = 0;
//         int rows = rows_per_process;
//         Mat subimage = image.rowRange(offset, offset + rows);
//         GaussianBlur(subimage, subimage, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);

//         // Receive processed parts from other processes
//         for (int i = 1; i < size; i++) {
//             int offset = i * rows_per_process;
//             int rows = (i == size - 1) ? image.rows - offset : rows_per_process;
//             MPI_Recv(image.ptr(offset), rows * image.step, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         }

//         // Save output image
//         std::string output_path = "blurred_" + std::to_string(kernel_size) + "_" + img_path;
//         imwrite(output_path, image);
//         std::cout << "Blurred image saved to " << output_path << "\n";
//     } else {
//         // Receive image data from root process
//         int rows;
//         MPI_Recv(&rows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         Mat subimage(rows, 1, CV_8UC3);
//         MPI_Recv(subimage.ptr(), rows * subimage.step, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//         // Process part of image
//         GaussianBlur(subimage, subimage, Size(kernel_size, kernel_size), 0, 0, BORDER_DEFAULT);

//         // Send processed part back to root process
//         MPI_Send(subimage.ptr(), rows * subimage.step, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
//     }

//     MPI_Finalize();
//     return 0;
// }



//////////////////////////////////////////////////////////////////////


// #include <iostream>
// #include <opencv2/opencv.hpp>
// #include <mpi.h>

// using namespace std;
// using namespace cv;

// int main(int argc, char** argv) {
//     int rank, size;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     // Get image path and kernel size from user
//     string img_path;
//     int kernel_size;
//     if (rank == 0) {
//         cout << "Enter image path: ";
//         cin >> img_path;
//         cout << "Enter kernel size: ";
//         cin >> kernel_size;
//     }

//     // Broadcast image path and kernel size to all processes
//     MPI_Bcast(&img_path, img_path.size(), MPI_CHAR, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Load image
//     Mat img = imread(img_path);

//     // Calculate number of rows per process
//     int rows_per_process = img.rows / size;

//     // Split image among processes
//     Mat img_partial(rows_per_process, img.cols, CV_8UC3);
//     MPI_Scatter(img.data, rows_per_process * img.cols * 3, MPI_BYTE,
//                 img_partial.data, rows_per_process * img.cols * 3, MPI_BYTE,
//                 0, MPI_COMM_WORLD);

//     // Blur image
//     Mat img_blur_partial;
//     GaussianBlur(img_partial, img_blur_partial, Size(kernel_size, kernel_size), 0);

//     // Gather blurred image data from all processes
//     Mat img_blur;
//     if (rank == 0) {
//         img_blur.create(img.rows, img.cols, CV_8UC3);
//     }
//     MPI_Gather(img_blur_partial.data, rows_per_process * img.cols * 3, MPI_BYTE,
//                img_blur.data, rows_per_process * img.cols * 3, MPI_BYTE,
//                0, MPI_COMM_WORLD);

//     // Display blurred image on rank 0 process
//     if (rank == 0) {
//         imshow("Blurred Image", img_blur);
//         waitKey(0);
//     }

//     MPI_Finalize();
//     return 0;
// }

////////////////////////////////////////////////////////////////////

// #include <iostream>
// #include <cstring>
// #include <mpi.h>
// #include <opencv2/opencv.hpp>

// using namespace std;
// using namespace cv;

// int main(int argc, char** argv)
// {
//     int rank, size;
//     Mat image;
//     int kernel_size;

//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (rank == 0) {
//         char img_path[100];
//         cout << "Enter image path: ";
//         cin >> img_path;
//         image = imread(img_path, IMREAD_COLOR);
//         if (image.empty()) {
//             cerr << "Error: Image not found or cannot be read" << endl;
//             MPI_Finalize();
//             return 1;
//         }
//         cout << "Enter kernel size: ";
//         cin >> kernel_size;
//     }

//     // Broadcast image path and kernel size to all processes
//     int len = 0;
//     if (rank == 0) {
//         len = strlen(argv[1]) + 1;
//     }
//     MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     char* img_path_c = new char[len];
//     if (rank == 0) {
//         strcpy(img_path_c, argv[1]);
//     }
//     MPI_Bcast(img_path_c, len, MPI_CHAR, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Load image on all processes
//     if (rank != 0) {
//         image = imread(img_path_c, IMREAD_COLOR);
//         if (image.empty()) {
//             cerr << "Error: Image not found or cannot be read" << endl;
//             MPI_Finalize();
//             delete[] img_path_c;
//             return 1;
//         }
//     }

//     // Calculate number of rows to process on each process
//     int rows_per_process = image.rows / size;
//     int start_row = rank * rows_per_process;
//     int end_row = (rank == size - 1) ? image.rows : start_row + rows_per_process;

//     // Blur image
//     Mat blurred_image = image.clone();
//     for (int i = start_row; i < end_row; i++) {
//         for (int j = 0; j < image.cols; j++) {
//             // Calculate kernel bounds
//             int kernel_half = kernel_size / 2;
//             int x_min = max(j - kernel_half, 0);
//             int x_max = min(j + kernel_half, image.cols - 1);
//             int y_min = max(i - kernel_half, 0);
//             int y_max = min(i + kernel_half, image.rows - 1);

//             // Calculate average color for kernel
//             Vec3b sum = 0;
//             int count = 0;
//             for (int y = y_min; y <= y_max; y++) {
//                 for (int x = x_min; x <= x_max; x++) {
//                     sum += image.at<Vec3b>(y, x);
//                     count++;
//                 }
//             }
//             Vec3b average = sum / count;

//             // Set pixel color to average
//             blurred_image.at<Vec3b>(i, j) = average;
//         }
//     }

//     // Collect processed rows from all processes to process 0
//     if (rank != 0) {
//         MPI_Send(blurred_image.data + start_row * blurred_image.step, rows_per_process * blurred_image.step, MPI_BYTE, 0,0, MPI_COMM_WORLD);
// } else {
//     for (int i = 1; i < size; i++) {
//         int start_row_i = i * rows_per_process;
//         int rows_i = (i == size - 1) ? image.rows - start_row_i : rows_per_process;
//         MPI_Recv(blurred_image.data + start_row_i * blurred_image.step, rows_i * blurred_image.step, MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//     }
// }

// // Save blurred image on process 0
// if (rank == 0) {
//     imshow("Original Image", image);
//     imshow("Blurred Image", blurred_image);
//     waitKey(0);
// }

// MPI_Finalize();
// delete[] img_path_c;
// return 0;
// }


/////////////////////////////////////////////////////////////////////////////////////////////////

// #include <iostream>
// #include <mpi.h>
// #include <opencv2/opencv.hpp>

// using namespace cv;
// using namespace std;

// int main(int argc, char** argv) {
//     int rank, size;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (argc != 3) {
//         if (rank == 0) {
//             cout << "Usage: " << argv[0] << " <image_path> <kernel_size>" << endl;
//         }
//         MPI_Finalize();
//         return -1;
//     }

//     // Read image on process 0 and broadcast to other processes
//     Mat image, blurred_image;
//     int kernel_size;

//     if (rank == 0) {
//         image = imread(argv[1], IMREAD_COLOR);
//         if (image.empty()) {
//             cout << "Error: Could not read image file" << endl;
//             MPI_Finalize();
//             return -1;
//         }

//         kernel_size = atoi(argv[2]);
//         if (kernel_size < 1 || kernel_size % 2 == 0) {
//             cout << "Error: Invalid kernel size" << endl;
//             MPI_Finalize();
//             return -1;
//         }

//         blurred_image.create(image.size(), image.type());
//     }

//     MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     int rows_per_process = image.rows / size;
//     int start_row = rank * rows_per_process;
//     int end_row = (rank == size - 1) ? image.rows : (rank + 1) * rows_per_process;

//     Mat image_part = image(Range(start_row, end_row), Range::all());

//     GaussianBlur(image_part, image_part, Size(kernel_size, kernel_size), 0, 0);

//     MPI_Gather(image_part.data, image_part.total() * image_part.elemSize(), MPI_BYTE,
//                blurred_image.data, image_part.total() * image_part.elemSize(), MPI_BYTE,
//                0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         imshow("Original Image", image);
//         imshow("Blurred Image", blurred_image);
//         waitKey(0);
//     }

//     MPI_Finalize();
//     return 0;
// }


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #include <mpi.h>
// #include <iostream>
// #include <fstream>
// #include <vector>

// using namespace std;

// int main(int argc, char** argv) {
//   // Initialize MPI.
//   MPI_Init(&argc, &argv);

//   // Get the number of processes.
//   int num_processes = MPI_COMM_WORLD.Get_size();

//   // Get the rank of the current process.
//   int rank = MPI_COMM_WORLD.Get_rank();

//   // Check if the number of processes is 1.
//   if (num_processes == 1) {
//     cout << "Error: The number of processes must be greater than 1." << endl;
//     MPI_Finalize();
//     return 1;
//   }

//   // Get the path of the image.
//   string image_path;
//   if (argc == 2) {
//     image_path = argv[1];
//   } else {
//     cout << "Error: The path of the image must be specified." << endl;
//     MPI_Finalize();
//     return 1;
//   }

//   // Get the kernel size.
//   int kernel_size;
//   if (argc == 3) {
//     kernel_size = atoi(argv[2]);
//   } else {
//     cout << "Error: The kernel size must be specified." << endl;
//     MPI_Finalize();
//     return 1;
//   }

//   // Read the image.
//   vector<vector<int>> image;
//   {
//     ifstream infile(image_path);
//     if (!infile.is_open()) {
//       cout << "Error: Could not open the image file." << endl;
//       MPI_Finalize();
//       return 1;
//     }

//     int width, height;
//     infile >> width >> height;

//     image.resize(height);
//     for (int i = 0; i < height; i++) {
//       image[i].resize(width);
//       for (int j = 0; j < width; j++) {
//         infile >> image[i][j];
//       }
//     }
//   }

//   // Calculate the blurred image.
//   vector<vector<int>> blurred_image(height);
//   for (int i = 0; i < height; i++) {
//     blurred_image[i].resize(width);
//     for (int j = 0; j < width; j++) {
//       int sum = 0;
//       for (int k = -kernel_size / 2; k <= kernel_size / 2; k++) {
//         for (int l = -kernel_size / 2; l <= kernel_size / 2; l++) {
//           if (i + k < 0 || i + k >= height || j + l < 0 || j + l >= width) {
//             continue;
//           }
//           sum += image[i + k][j + l];
//         }
//       }
//       blurred_image[i][j] = sum / (kernel_size * kernel_size);
//     }
//   }

//   // Write the blurred image.
//   ofstream outfile("blurred_image.pgm");
//   outfile << "P5" << endl;
// //   outfile << width << " " << height << endl;
//   outfile << 255 << endl;
//   for (int i = 0; i < height; i++) {
//     for (int j = 0; j < width; j++) {
//       outfile << blurred_image[i][j] << " ";
//     }
//     outfile << endl;
//   }

//   // Finalize MPI.
//   MPI_Finalize();

//   return 0;
// }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #include <mpi.h>
// #include <iostream>
// #include <fstream>
// #include <vector>

// using namespace std;

// int main(int argc, char** argv) {

//   // Initialize MPI.
//   MPI_Init(&argc, &argv);

//   // Get the number of processes.
//   int num_processes;
//     MPI_Comm_size(MPI_COMM_WORLD, &num_processes);

  

//   // Get the rank of the current process.
//   int rank;
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//   // Get the path of the image.
//   string image_path = argv[1];

//   // Get the kernel size.
//   int kernel_size = atoi(argv[2]);

//   // Read the image.
//   vector<vector<unsigned char>> image;
//   ifstream infile(image_path);
//   if (!infile.is_open()) {
//     cout << "Error opening image file." << endl;
//     return 1;
//   }
//   for (int y = 0; y < image.size(); y++) {
//     vector<unsigned char> row;
//     for (int x = 0; x < image[y].size(); x++) {
//       unsigned char value;
//       infile >> value;
//       row.push_back(value);
//     }
//     image.push_back(row);
//   }
//   infile.close();

//   // Create a blurred image.
//   vector<vector<unsigned char>> blurred_image(image.size(), vector<unsigned char>(image[0].size()));
//   for (int y = 0; y < image.size(); y++) {
//     for (int x = 0; x < image[y].size(); x++) {
//       unsigned char sum = 0;
//       for (int i = -kernel_size / 2; i <= kernel_size / 2; i++) {
//         for (int j = -kernel_size / 2; j <= kernel_size / 2; j++) {
//           if (y + i < 0 || y + i >= image.size() || x + j < 0 || x + j >= image[0].size()) {
//             continue;
//           }
//           sum += image[y + i][x + j];
//         }
//       }
//       blurred_image[y][x] = sum / (kernel_size * kernel_size);
//     }
//   }

//   // Write the blurred image to a file.
//   ofstream outfile("blurred_image.png");
//   for (int y = 0; y < blurred_image.size(); y++) {
//     for (int x = 0; x < blurred_image[y].size(); x++) {
//       outfile << blurred_image[y][x];
//     }
//     outfile << endl;
//   }
//   outfile.close();

//   // Finalize MPI.
//   MPI_Finalize();

//   return 0;
// }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #include <iostream>
// #include <mpi.h>
// #include <opencv2/opencv.hpp>

// using namespace cv;
// using namespace std;

// int main(int argc, char** argv) {
//     int rank, size;
//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (argc != 3) {
//         if (rank == 0) {
//             cout << "Usage: " << argv[0] << " <image_path> <kernel_size>" << endl;
//         }
//         MPI_Finalize();
//         return -1;
//     }

//     // Read image on process 0 and broadcast to other processes
//     Mat image, blurred_image;
//     int kernel_size;

//     if (rank == 0) {
//         image = imread(argv[1], IMREAD_COLOR);
//         if (image.empty()) {
//             cout << "Error: Could not read image file" << endl;
//             MPI_Finalize();
//             return -1;
//         }

//         kernel_size = atoi(argv[2]);
//         if (kernel_size < 1 || kernel_size % 2 == 0) {
//             cout << "Error: Invalid kernel size" << endl;
//             MPI_Finalize();
//             return -1;
//         }

//         blurred_image.create(image.size(), image.type());
//     }

//     MPI_Bcast(&kernel_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     int rows_per_process = image.rows / size;
//     int start_row = rank * rows_per_process;
//     int end_row = (rank == size - 1) ? image.rows : (rank + 1) * rows_per_process;

//     Mat image_part = image(Range(start_row, end_row), Range::all());

//     GaussianBlur(image_part, image_part, Size(kernel_size, kernel_size), 0, 0);

//     MPI_Gather(image_part.data, image_part.total() * image_part.elemSize(), MPI_BYTE,
//                blurred_image.data, image_part.total() * image_part.elemSize(), MPI_BYTE,
//                0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         imshow("Original Image", image);
//         imshow("Blurred Image", blurred_image);
//         waitKey(0);
//     }

//     MPI_Finalize();
//     return 0;
// }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #include <mpi.h>
// #include <opencv2/opencv.hpp>

// using namespace cv;

// int main(int argc, char** argv) {
//     int rank, size;
//     Mat img, blurred_img;
//     double start, end;

//     MPI_Init(&argc, &argv);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (rank == 0) {
//         // Load the image
//         img = imread("milkyWay.png");
//         if (img.empty()) {
//             std::cerr << "Failed to read image" << std::endl;
//             MPI_Abort(MPI_COMM_WORLD, 1);
//         }

//         start = MPI_Wtime();
//     }

//     // Broadcast the image dimensions
//     int rows, cols;
//     if (rank == 0) {
//         rows = img.rows;
//         cols = img.cols;
//     }
//     MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Split the rows among processes
//     int rows_per_process = rows / size;
//     int start_row = rank * rows_per_process;
//     int end_row = (rank == size - 1) ? rows : (rank + 1) * rows_per_process;

//     // Allocate memory for the local portion of the image
//     Mat local_img(end_row - start_row, cols, CV_8UC3);

//     // Scatter the image data among processes
//     MPI_Scatter(img.data, (rows_per_process * cols * 3), MPI_BYTE,
//                 local_img.data, (rows_per_process * cols * 3), MPI_BYTE,
//                 0, MPI_COMM_WORLD);

//     // Apply Gaussian blur filter to the local portion of the image
//     GaussianBlur(local_img, local_img, Size(15, 15), 0);

//     // Gather the blurred image data back to the root process
//     if (rank == 0) {
//         blurred_img = Mat(rows, cols, CV_8UC3);
//     }
//     MPI_Gather(local_img.data, (rows_per_process * cols * 3), MPI_BYTE,
//                blurred_img.data, (rows_per_process * cols * 3), MPI_BYTE,
//                0, MPI_COMM_WORLD);

//     if (rank == 0) {
//         end = MPI_Wtime();
//         std::cout << "Elapsed time: " << end - start << " seconds" << std::endl;

//         // Save the blurred image
//         imwrite("blurred.jpg", blurred_img);
//     }

//     MPI_Finalize();

//     return 0;
// }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string>
// #include <chrono>
// #include <mpi.h>

// using namespace std;
// using namespace cv;
// using namespace chrono;


// int main() {

//     // Initialize MPI
//     MPI_Init(NULL, NULL);
//     int world_size;
//     MPI_Comm_size(MPI_COMM_WORLD, &world_size);
//     int world_rank;
//     MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

//     // Read the image file
//     Mat image;
//     if (world_rank == 0) {
//         image = imread("lena.png");
//         // Check if the image was read successfully
//     if (image.empty()) {
//         cerr << "Error: Could not read image file." << endl;
//         return 1;
//     }
//     }

    

//     // Ask user for kernel size 
//     int Ksize;
//     if (world_rank == 0) {
//         cout << "Enter the kernel size: ";
//         cin >> Ksize;
//     }

//     // Broadcast the kernel size to all ranks
//     MPI_Bcast(&Ksize, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Create a kernel for blurring
//     int k = Ksize / 2;
//     Mat kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);
    
//     // Create a copy of the image to hold the blurred image
//     Mat blurred_image(image.size(), image.type());
    
//     // start timer
//     auto start_time = high_resolution_clock::now();

//     // Apply the kernel to each pixel in the image
//     int num_threads = world_size - 1;
//     int thread_size = image.rows / num_threads;
//     int start_row = thread_size * world_rank;
//     int end_row = min(start_row + thread_size, image.rows);
//     for (int i = start_row; i < end_row; i++) {
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

//     // end timer
//     auto end_time = high_resolution_clock::now();

//     // show both clear and blurred image
//     if (world_rank == 0) {
//         namedWindow("Original image", WINDOW_NORMAL);
//         namedWindow("Filtered image", WINDOW_NORMAL);
//         imshow("Original image", image);
//         imshow("Filtered image", blurred_image);
//         waitKey(0);
//     }

//     // Finalize MPI
//     MPI_Finalize();

//     return 0;
// }

//////////////////////////////////////////////////////////work on one process/////////////////////////////////////////////////////////////////////



// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string>
// #include <chrono>
// #include <mpi.h>

// using namespace std;
// using namespace cv;
// using namespace chrono;

// int main(int argc, char* argv[]) {


//     // cout << "Enter the image name:1 ";
//     // int input = cin.get();

//     // Initialize MPI
//     MPI_Init(&argc, &argv);

//     // Get the rank and size of the MPI processes
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     Mat image = imread("lena.png");

//     // Ask user for image path (only rank 0 does this)
//     // string imagePath;
//     // if (rank == 0) {
//     //     cout << "Enter the image name: ";
//     //     cin >> imagePath;
//     // }

//     // Broadcast the size of imagePath to all processes
//     // int imagePathSize = 0;
//     // if (rank == 0) {
//     //     imagePathSize = imagePath.size();
//     // }
//     // MPI_Bcast(&imagePathSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Resize the imagePath string on each process to accommodate the received size
//     // imagePath.resize(imagePathSize);

//     // // Broadcast the imagePath string to all processes
//     // MPI_Bcast(&imagePath[0], imagePathSize, MPI_CHAR, 0, MPI_COMM_WORLD);

//     // Load the image (only rank 0 does this)
//     // Mat image;
//     // if (rank == 0) {
//     //     image = imread(imagePath);

//     //     // Check the entered image path
//     //     if (image.empty()) {
//     //         cerr << "Error: Could not read image file." << endl;
//     //         MPI_Finalize();
//     //         return 1;
//     //     }
//     // }

//     // Broadcast the image dimensions to all processes
//     // int imageRows, imageCols;
//     // if (rank == 0) {
//     //     imageRows = image.rows;
//     //     imageCols = image.cols;
//     // }
//     // MPI_Bcast(&imageRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     // MPI_Bcast(&imageCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Ask user for kernel size (only rank 0 does this)
//     int Ksize;
//     if (rank == 0) {
//         cout << "Enter the kernel size: ";
//         cin >> Ksize;

//         // Check that the kernel number entered is odd
//         while (Ksize < 3 || Ksize % 2 != 1) {
//             cout << "The kernel size must be an odd number greater than or equal to 3. Enter the kernel size: ";
//             cin >> Ksize;
//         }
//     }

//     // Broadcast the kernel size to all processes
//     MPI_Bcast(&Ksize, 1, MPI_INT, 0, MPI_COMM_WORLD);


//     // Create a kernel for blurring (only rank 0 does this)
//     Mat kernel;
//     // if (rank == 0) {
//     //     int k = Ksize / 2;
//     //     kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);
//     // }
//         kernel = Mat::ones(Ksize, Ksize, CV_32F) / (float)(Ksize * Ksize);

//     // Calculate the chunk size for each process
//     int chunkSize = (image.rows - Ksize + 1) / size;
//     int remainder = (image.rows - Ksize + 1) % size;
//     int startRow = rank * chunkSize;
//     int endRow = startRow + chunkSize - 1;

//     // Adjust the endRow for the last process
//     if (rank == size - 1) {
//         endRow += remainder;
//     }

//     // Create a copy of the image chunk to hold the blurred image chunk
//     Mat blurred_image = image(Rect(0, startRow, image.cols, endRow - startRow + 1)).clone();

//     // Start the timer (only rank 0 does this)
//     double start_time;
//     if (rank == 0) {
//         start_time = MPI_Wtime();
//     }

//     cout<< "herrere"<< endl;
//     // Apply the kernel to each pixel in the image chunk
//     for (int i = 0; i < blurred_image.rows; i++) {
//         for (int j = Ksize / 2; j < blurred_image.cols - Ksize / 2; j++) {
//             float sum_r = 0;
//             float sum_g = 0;
//             float sum_b = 0;
//             for (int x = -Ksize / 2; x <= Ksize / 2; x++) {
//                 for (int y = -Ksize / 2; y <= Ksize / 2; y++) {
//                     Vec3b pixel = image.at<Vec3b>(startRow + i + x, j + y);
//                     sum_b += pixel[0] * kernel.at<float>(Ksize / 2 + x, Ksize / 2 + y);
//                     sum_g += pixel[1] * kernel.at<float>(Ksize / 2 + x, Ksize / 2 + y);
//                     sum_r += pixel[2] * kernel.at<float>(Ksize / 2 + x, Ksize / 2 + y);
//                 }
//             }
//             Vec3b new_pixel(sum_b, sum_g, sum_r);
//             blurred_image.at<Vec3b>(i, j) = new_pixel;
//         }
//     }
//     cout<< "herrere2"<< endl;

//     // Gather the blurred image chunks from all processes to reconstruct the complete blurred image (only rank 0 does this)
//     if (rank == 0) {
//         vector<Mat> blurred_chunks(size);
//         blurred_chunks[0] = blurred_image;
//         for (int i = 1; i < size; i++) {
//             blurred_chunks[i] = Mat(chunkSize, image.cols, CV_8UC3);
//             MPI_Recv(blurred_chunks[i].data, blurred_chunks[i].total() * blurred_chunks[i].elemSize(), MPI_BYTE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         }

//         // Concatenate the blurred image chunks
//         vconcat(blurred_chunks, blurred_image);
//     }
//     else {
//         // Send the blurred image chunk to rank 0
//         MPI_Send(blurred_image.data, blurred_image.total() * blurred_image.elemSize(), MPI_BYTE, 0, 0, MPI_COMM_WORLD);
//     }

//     // End the timer (only rank 0 does this)
//     double end_time;
//     if (rank == 0) {
//         end_time = MPI_Wtime();
//     }

//     // Rank 0 displays the original and blurred images and prints the execution time
//     if (rank == 0) {
//         // Display the images
//         namedWindow("Original image", WINDOW_NORMAL);
//         namedWindow("Filtered image", WINDOW_NORMAL);
//         imshow("Original image", image);
//         imshow("Filtered image", blurred_image);

//         // Show the execution time in seconds
//         cout << "Execution time: " << end_time - start_time << " seconds\n";
//     }

//     // Wait for key press (all processes)
//     waitKey(0);

//     // Finalize MPI
//     MPI_Finalize();

//     return 0;
// }

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #include <opencv2/opencv.hpp>
// #include <iostream>
// #include <string>
// #include <chrono>
// #include <mpi.h>

// using namespace std;
// using namespace cv;
// using namespace chrono;

// int main(int argc, char* argv[]) {
//     // Initialize MPI
//     MPI_Init(&argc, &argv);

//     // Get the rank and size of the MPI processes
//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     // Ask user for image path (only rank 0 does this)
//     string imagePath;
//     if (rank == 0) {
//         cout << "Enter the image name: ";
//         cin >> imagePath;
//     }

//     // Broadcast the size of imagePath to all processes
//     int imagePathSize = 0;
//     if (rank == 0) {
//         imagePathSize = imagePath.size();
//     }
//     MPI_Bcast(&imagePathSize, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Resize the imagePath string on each process to accommodate the received size
//     imagePath.resize(imagePathSize + 1); // Adjust size to include null character

//     // Broadcast the imagePath string to all processes
//     MPI_Bcast(&imagePath[0], imagePathSize + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

//     // Remove the null character from imagePath on each process
//     imagePath.resize(imagePathSize);

//     // Load the image (only rank 0 does this)
//     Mat image;
//     if (rank == 0) {
//         image = imread(imagePath);

//         // Check the entered image path
//         if (image.empty()) {
//             cerr << "Error: Could not read image file." << endl;
//             MPI_Finalize();
//             return 1;
//         }
//     }

//     // Broadcast the image dimensions to all processes
//     int imageRows, imageCols;
//     if (rank == 0) {
//         imageRows = image.rows;
//         imageCols = image.cols;
//     }
//     MPI_Bcast(&imageRows, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&imageCols, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Calculate the chunk size for each process
//     int chunkSize = imageRows / size;
//     int remainder = imageRows % size;
//     int startRow = rank * chunkSize;
//     int endRow = startRow + chunkSize - 1;

//     // Adjust the endRow for the last process
//     if (rank == size - 1) {
//         endRow += remainder;
//     }

//     // Create a copy of the image chunk to hold the blurred image chunk
//     Mat blurredImageChunk = Mat(endRow - startRow + 1, imageCols, CV_8UC3);

//     // Start the timer (only rank 0 does this)
//     double start_time;
//     if (rank == 0) {
//         start_time = MPI_Wtime();
//     }

//     // Apply the kernel to each pixel in the image chunk
//     for (int i = startRow; i <= endRow; i++) {
//         for (int j = 0; j < imageCols; j++) {
//             float sumR = 0.0, sumG = 0.0, sumB = 0.0;
//             int count = 0;

//             // Apply the kernel to the pixel and its neighbors
//             for (int x = -1; x <= 1; x++) {
//                 for (int y = -1; y <= 1; y++) {
//                     if (i + x >= 0 && i + x < imageRows && j + y >= 0 && j + y < imageCols) {
//                         Vec3b pixel = image.at<Vec3b>(i + x, j + y);
//                         sumB += pixel[0];
//                         sumG += pixel[1];
//                         sumR += pixel[2];
//                         count++;
//                     }
//                 }
//             }

//             // Compute the average value for the blurred pixel
//             Vec3b blurredPixel(static_cast<uchar>(sumB / count), static_cast<uchar>(sumG / count), static_cast<uchar>(sumR / count));
// blurredImageChunk.at<Vec3b>(i - startRow, j) = blurredPixel;
// }
// }
// // Gather all image chunks at rank 0
// Mat blurredImage;
// if (rank == 0) {
//     blurredImage = Mat(imageRows, imageCols, CV_8UC3);
// }
// MPI_Gather(blurredImageChunk.data, blurredImageChunk.total() * blurredImageChunk.elemSize(), MPI_BYTE,
//            blurredImage.data, blurredImageChunk.total() * blurredImageChunk.elemSize(), MPI_BYTE, 0, MPI_COMM_WORLD);

// // Stop the timer (only rank 0 does this)
// double end_time;
// if (rank == 0) {
//     end_time = MPI_Wtime();
//     double elapsed_time = end_time - start_time;
//     cout << "Elapsed Time: " << elapsed_time << " seconds" << endl;

//     // Display the blurred image
//     namedWindow("Blurred Image", WINDOW_NORMAL);
//     imshow("Blurred Image", blurredImage);
//     waitKey(0);
// }

// // Finalize MPI
// MPI_Finalize();

// return 0;
// }
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// #include <iostream>
// #include <mpi.h>
// #include <opencv2/opencv.hpp>

// using namespace cv;

// // Function to blur the image
// Mat blurImage(const Mat& inputImage)
// {
//     Mat blurredImage;
//     blur(inputImage, blurredImage, Size(15, 15));
//     return blurredImage;
// }

// int main(int argc, char** argv)
// {
//     MPI_Init(&argc, &argv);

//     int rank, size;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &size);

//     if (rank == 0)
//     {
//         // Read the clear image using OpenCV
//         Mat clearImage = imread("lena.png", IMREAD_COLOR);
//         if (clearImage.empty())
//         {
//             std::cout << "Failed to read the clear image." << std::endl;
//             MPI_Finalize();
//             return -1;
//         }

//         // Split the image into equal-sized chunks
//         int rowsPerProcess = clearImage.rows / size;
//         int remainingRows = clearImage.rows % size;
//         int startRow = 0;

//         for (int i = 1; i < size; i++)
//         {
//             int numRows = rowsPerProcess;
//             if (i == size - 1)
//                 numRows += remainingRows;

//             MPI_Send(&startRow, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
//             MPI_Send(&numRows, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
//             MPI_Send(clearImage.ptr(startRow), numRows * clearImage.cols * clearImage.channels(), MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD);

//             startRow += numRows;
//         }

//         // Process the remaining chunk of the image in the master process
//         Mat blurredImage = blurImage(clearImage.rowRange(startRow, startRow + rowsPerProcess + remainingRows));

//         // Receive processed image chunks from worker processes
//         for (int i = 1; i < size; i++)
//         {
//             int startRow;
//             MPI_Recv(&startRow, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//             int numRows = rowsPerProcess;
//             if (i == size - 1)
//                 numRows += remainingRows;

//             MPI_Recv(blurredImage.ptr(startRow), numRows * clearImage.cols * clearImage.channels(), MPI_UNSIGNED_CHAR, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         }

//         // Save the blurred image
//         imwrite("blurred_image.png", blurredImage);

//         std::cout << "Image successfully blurred." << std::endl;
//     }
//     else
//     {
//         // Receive image chunk from master process
//         int startRow, numRows;
//         MPI_Recv(&startRow, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//         MPI_Recv(&numRows, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//         Mat imageChunk(numRows, MPI_ANY_TAG, CV_8UC3);
//         MPI_Recv(imageChunk.ptr(), numRows * imageChunk.cols * imageChunk.channels(), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

//         // Process the image chunk
//         Mat blurredChunk = blurImage(imageChunk);

//         // Send the processed image chunk back to the master process
//         MPI_Send(&startRow,1, MPI_INT, 0, 0, MPI_COMM_WORLD);
//         MPI_Send(blurredChunk.ptr(), numRows * blurredChunk.cols * blurredChunk.channels(), MPI_UNSIGNED_CHAR, 0, 0, MPI_COMM_WORLD);
//     }
//     MPI_Finalize();
//     return 0;
// }
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// #include <mpi.h>
// #include <opencv2/opencv.hpp>

// using namespace cv;

// void blurImage(Mat& image, int kernelSize, Mat& kernel)
// {
//     Mat blurred_Image=image.clone();
//     int padding = kernelSize / 2;
//     Mat paddedImage;
//     copyMakeBorder(image, paddedImage, padding, padding, padding, padding, BORDER_REPLICATE);

//     int rows = image.rows;
//     int cols = image.cols;
//     int channels = image.channels();

//     // for (int i = padding; i < image.rows ; i++) {
//     //     for (int j = padding; j < image.cols ; j++) {
//     //         float sum_r = 0;
//     //         float sum_g = 0;
//     //         float sum_b = 0;
//     //         for (int x = -padding; x <= padding; x++) {
//     //             for (int y = -padding; y <= padding; y++) {
//     //                 Vec3b pixel = blurred_Image.at<Vec3b>(i + x, j + y);
//     //                 sum_b += pixel[0] * kernel.at<float>(padding + x, padding + y);
//     //                 sum_g += pixel[1] * kernel.at<float>(padding + x, padding + y);
//     //                 sum_r += pixel[2] * kernel.at<float>(padding + x, padding + y);
//     //             }
//     //         }
//     //         Vec3b new_pixel(sum_b, sum_g, sum_r);
//     //         image.at<Vec3b>(i, j) = new_pixel;
//     //     }
//     // }
//     // Apply the kernel to each pixel in the image
//     for (int i = padding; i < image.rows ; i++) {
//         for (int j = padding; j < image.cols -padding; j++) {
//             float sum_r = 0;
//             float sum_g = 0;
//             float sum_b = 0;
//             for (int x = -padding; x <= padding; x++) {
//                 for (int y = -padding; y <= padding; y++) {
//                     Vec3b pixel = blurred_Image.at<Vec3b>(i + x, j + y);
//                     sum_b += pixel[0] * kernel.at<float>(padding + x, padding+ y);
//                     sum_g += pixel[1] * kernel.at<float>(padding + x, padding+ y);
//                     sum_r += pixel[2] * kernel.at<float>(padding + x, padding+ y);
//                 }
//             }
//             Vec3b new_pixel(sum_b, sum_g, sum_r);
//             image.at<Vec3b>(i, j) = new_pixel;
//         }
//     }
// }

// int main(int argc, char** argv)
// {
//     MPI_Init(NULL, NULL);

//     int numProcesses, rank;
//     MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);

//     if (argc < 3) {
//         if (rank == 0) {
//             std::cerr << "Usage: " << argv[0] << " <image_path> <kernel_size>" << std::endl;
//         }
//         MPI_Finalize();
//         return 1;
//     }

//     const char* imagePath = argv[1];
//     int kernelSize = atoi(argv[2]);

//     Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize * kernelSize);

//     Mat image;

//     if (rank == 0) {
//         // Read the image
//         image = imread(imagePath, IMREAD_COLOR);
//         if (image.empty()) {
//             std::cerr << "Failed to read the image: " << imagePath << std::endl;
//             MPI_Finalize();
//             return 1;
//         }
//     }

//     // Broadcast image size information to all processes
//     int rows = 0, cols = 0;
//     if (rank == 0) {
//         rows = image.rows;
//         cols = image.cols;
//     }
//     MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
//     MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

//     // Split the image into chunks and distribute the rows among processes
//     int chunkSize = (rows + numProcesses - 1) / numProcesses;  // ceil(rows / numProcesses)
//     int startRow = rank * chunkSize;
//     int endRow = std::min(startRow + chunkSize, rows);

//     if (rank != 0) {
//         // Allocate memory for the received chunk
//         image.create(chunkSize, cols, CV_8UC3);
//     }

//     // Scatter image data to all processes
//     MPI_Scatter(image.data, chunkSize * cols * 3, MPI_UNSIGNED_CHAR, image.data, chunkSize * cols * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

//     // Process the chunk
//     blurImage(image, kernelSize, kernel);

//     // Gather the processed chunks back to the root process
//    // Gather the processed chunks back to the root process
// MPI_Gather(image.data, chunkSize * cols * 3, MPI_UNSIGNED_CHAR, image.data, chunkSize * cols * 3, MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD);

// // Display the blurred image on the root process
// if (rank == 0) {
//     // Create a blank image to hold the final result
//     Mat blurredImage(rows, cols, CV_8UC3);

//     // Copy the processed chunks to the final image
//     memcpy(blurredImage.data, image.data, rows * cols * 3);

//     // Display the blurred image
//     namedWindow("Blurred Image", WINDOW_NORMAL);
//     imshow("Blurred Image", blurredImage);
//     waitKey(0);
// }

// MPI_Finalize();
// return 0;
// }



////////////////////////////////////////////////////working code////////////////////////////////////////////////////////////////

#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace cv;

void blurImage(Mat& image, int kernelSize, Mat& kernel)
{
    Mat blurred_Image=image.clone();
    int padding = kernelSize / 2;
    Mat paddedImage;
    copyMakeBorder(image, paddedImage, padding, padding, padding, padding, BORDER_REPLICATE);

    int rows = image.rows;
    int cols = image.cols;
    int channels = image.channels();

    // Apply the kernel to each pixel in the image
    for (int i = padding; i < image.rows ; i++) {
        for (int j = padding; j < image.cols -padding; j++) {
            float sum_r = 0;
            float sum_g = 0;
            float sum_b = 0;
            for (int x = -padding; x <= padding; x++) {
                for (int y = -padding; y <= padding; y++) {
                    Vec3b pixel = blurred_Image.at<Vec3b>(i + x, j + y);
                    sum_b += pixel[0] * kernel.at<float>(padding + x, padding+ y);
                    sum_g += pixel[1] * kernel.at<float>(padding + x, padding+ y);
                    sum_r += pixel[2] * kernel.at<float>(padding + x, padding+ y);
                }
            }
            Vec3b new_pixel(sum_b, sum_g, sum_r);
            image.at<Vec3b>(i, j) = new_pixel;
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(NULL, NULL);

    int numProcesses, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc < 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <image_path> <kernel_size>" << std::endl;
        }
        MPI_Finalize();
        return 0;
    }

    const char* imagePath = argv[1];
    int kernelSize = atoi(argv[2]);

    Mat kernel = Mat::ones(kernelSize, kernelSize, CV_32F) / (float)(kernelSize * kernelSize);

    Mat image, paddedImage;

    if (rank == 0) {
        // Read the image
        image = imread(imagePath, IMREAD_COLOR);
        if (image.empty()) {
            std::cerr << "Failed to read the image: " << imagePath << std::endl;
            MPI_Finalize();
            return 0;
        }
    }
    double start_time;
    if (rank == 0) {
        start_time = MPI_Wtime();
    }
    // Broadcast image size information to all processes
    int rows = 0, cols = 0;
    if (rank == 0) {
        rows = image.rows;
        cols = image.cols;
    }
    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Split the image into chunks and distribute the rows among processes
    int chunkSize = rows / numProcesses;  // ceil(rows / numProcesses)
    int extraRows = rows % numProcesses;
    
    int paddingSize = kernelSize / 2;
    int shadowSize = kernelSize / 2;
    int k = kernelSize / 2;
    
    float *flattened_image, *flattened_image_chunk;
    if (rank == 0)
    {
        // Allocate memory for the padded image
        copyMakeBorder(image, paddedImage, paddingSize, paddingSize, paddingSize, paddingSize, BORDER_REPLICATE);
        flattened_image = new float[paddedImage.rows * paddedImage.cols * paddedImage.channels()];

        // Flatten the image
        for(int i = 0; i < paddedImage.rows; i++)
        {
            for(int j = 0; j < paddedImage.cols; j++)
            {
                for(int k = 0; k < paddedImage.channels(); k++)
                {
                    flattened_image[(i * paddedImage.cols * paddedImage.channels()) + (j * paddedImage.channels()) + k] = paddedImage.at<Vec3b>(i, j)[k];
                }
            }
        }

        image.deallocate();
    }

    // Allocate memory for the received chunk
    image.create(chunkSize + (2* shadowSize), cols + (2 * paddingSize), CV_8UC3);
    flattened_image_chunk = new float[image.rows * image.cols * image.channels()];

    // Scatter the flattened image to all processes
	int *counts = 0, *displacements = 0;
	if (rank == 0)
	{
		counts = new int[numProcesses];
		displacements = new int[numProcesses];

		// Calculate the counts and displacement values
		for (int i = 0; i < numProcesses; i++)
		{
			counts[i] = (chunkSize + 2 * shadowSize) * (cols + 2 * paddingSize) * 3;
            displacements[i] = i * (chunkSize * (cols + 2 * paddingSize) * 3);
		}
	}

    // Scatter the flattened image
	MPI_Scatterv(flattened_image, counts, displacements, MPI_FLOAT, flattened_image_chunk, (chunkSize + 2 * shadowSize) * (cols + 2 * paddingSize) * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Convert the flattened image chunk to a Mat
    for(int i = 0; i < image.rows; i++)
    {
        for(int j = 0; j < image.cols; j++)
        {
            Vec3b new_pixel(0, 0, 0);
            for(int k = 0; k < image.channels(); k++)
            {
                new_pixel[k] = flattened_image_chunk[(i * image.cols * image.channels()) + (j * image.channels()) + k];
            }
            image.at<Vec3b>(i, j) = new_pixel;
        }
    }

    // Apply the filter
    Mat filteredChunk;
    filteredChunk.create(chunkSize, cols, CV_8UC3);
    for (int i = shadowSize; i < image.rows - shadowSize; i++) {
        for (int j = shadowSize; j < image.cols - shadowSize; j++) {
            float sum_r = 0;
            float sum_g = 0;
            float sum_b = 0;
            for (int x = -k; x <= k; x++) {
                for (int y = -k; y <= k; y++) {
                    Vec3b pixel = image.at<Vec3b>(i + x, j + y);
                    sum_b += pixel[0] * kernel.at<float>(k + x, k + y);
                    sum_g += pixel[1] * kernel.at<float>(k + x, k + y);
                    sum_r += pixel[2] * kernel.at<float>(k + x, k + y);
                }
            }
            Vec3b new_pixel(sum_b, sum_g, sum_r);
            filteredChunk.at<Vec3b>(i-k, j-k) = new_pixel;
        }
    }


    // Apply the filter to the rows that could not be sent to other processes (due to division of rows by numProcesses not being perfect)
    Mat extraChunk;
    int extraStartRow = chunkSize * numProcesses;
    if(rank == 0)
    {
        extraChunk.create(extraRows, cols, CV_8UC3);
        for(int i = 0; i < extraRows; i++)
        {
            for(int j = 0; j < cols; j++)
            {
                float sum_r = 0;
                float sum_g = 0;
                float sum_b = 0;
                for (int x = -k; x <= k; x++) {
                    for (int y = -k; y <= k; y++) {
                        Vec3b pixel = paddedImage.at<Vec3b>(extraStartRow + i + x, j + y);
                        sum_b += pixel[0] * kernel.at<float>(k + x, k + y);
                        sum_g += pixel[1] * kernel.at<float>(k + x, k + y);
                        sum_r += pixel[2] * kernel.at<float>(k + x, k + y);
                    }
                }
                Vec3b new_pixel(sum_b, sum_g, sum_r);
                extraChunk.at<Vec3b>(i, j) = new_pixel;
            }
        }
    }

    // Convert the filtered chunk to a flattened array
    float *flattened_filtered_chunk = new float[filteredChunk.rows * filteredChunk.cols * filteredChunk.channels()];
    for(int i = 0; i < filteredChunk.rows; i++)
    {
        for(int j = 0; j < filteredChunk.cols; j++)
        {
            for(int k = 0; k < filteredChunk.channels(); k++)
            {
                flattened_filtered_chunk[(i * filteredChunk.cols * filteredChunk.channels()) + (j * filteredChunk.channels()) + k] = filteredChunk.at<Vec3b>(i, j)[k];
            }
        }
    }

    if(rank == 0)
    {
        image.deallocate();
        image.create(rows, cols, CV_8UC3);
    }

    // // Gather the filtered chunks
    MPI_Gather(flattened_filtered_chunk, chunkSize * cols * 3, MPI_FLOAT, flattened_image, chunkSize * cols * 3, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // Convert the flattened image to a Mat
    if(rank == 0)
    {
        for(int i = 0; i < image.rows; i++)
        {
            for(int j = 0; j < image.cols; j++)
            {
                Vec3b new_pixel(0, 0, 0);
                for(int k = 0; k < image.channels(); k++)
                {
                    new_pixel[k] = flattened_image[(i * image.cols * image.channels()) + (j * image.channels()) + k];
                }
                image.at<Vec3b>(i, j) = new_pixel;
            }
        }

        for (size_t i = 0; i < extraRows; i++)
        {
            for (size_t j = 0; j < image.cols; j++)
            {
                image.at<Vec3b>(extraStartRow + i, j) = extraChunk.at<Vec3b>(i, j);
            }
        }
        
        double end_time = MPI_Wtime();
        std::cout << "Execution time: " << end_time - start_time << " seconds\n";
        
        
        imshow("Blurred Image", image);

        waitKey(0);
    }

    

MPI_Finalize();
return 0;
}


