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
