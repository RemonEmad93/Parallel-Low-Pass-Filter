from mpi4py import MPI
import numpy as np
import cv2

# Load the clear image using OpenCV
clear_image = cv2.imread('lena.png')

# Get the image dimensions
height, width, channels = clear_image.shape

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Determine the chunk size for each process
chunk_size = height // size
remainder = height % size

# Calculate the start and end indices for the current process
start_index = rank * chunk_size
end_index = start_index + chunk_size

# Adjust the end index for the last process
if rank == size - 1:
    end_index += remainder

# Divide the clear image into chunks and scatter them among processes
chunk = np.empty((chunk_size, width, channels), dtype=np.uint8)
comm.Scatter(clear_image[start_index:end_index], chunk)

# Apply the blurring operation to the chunk
blurred_chunk = cv2.GaussianBlur(chunk, (15, 15), 0)

# Gather all the blurred chunks at the root process
blurred_image = None
if rank == 0:
    blurred_image = np.empty_like(clear_image)
blurred_chunks = [blurred_chunk]
comm.Gatherv(blurred_chunk, [blurred_image, (chunk_size, width, channels), None, MPI.DOUBLE])

# Display or save the blurred image
if rank == 0:
    cv2.imshow('Blurred Image', blurred_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('blurred_image.png', blurred_image)
