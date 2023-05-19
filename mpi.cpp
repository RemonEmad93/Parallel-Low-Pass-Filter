#include <mpi.h>
#include <opencv2/opencv.hpp>

using namespace cv;

// void blurImage(Mat& image, int kernelSize, Mat& kernel)
// {
//     Mat blurred_Image=image.clone();
//     int padding = kernelSize / 2;
//     Mat paddedImage;
//     copyMakeBorder(image, paddedImage, padding, padding, padding, padding, BORDER_REPLICATE);

//     int rows = image.rows;
//     int cols = image.cols;
//     int channels = image.channels();

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

