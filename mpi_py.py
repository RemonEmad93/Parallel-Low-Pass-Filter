import cv2
from mpi4py import MPI

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Load the image
if rank == 0:
    img = cv2.imread('lena.png')

# Broadcast the image to all the processes
img = comm.bcast(img, root=0)

# Define the kernel size for the Gaussian blur
kernel_size = (31, 31)

# Calculate the start and end rows for each process
rows_per_process = img.shape[0] // size
start_row = rank * rows_per_process
end_row = start_row + rows_per_process

# Apply the Gaussian blur to the assigned rows
blurred_img = cv2.GaussianBlur(img[start_row:end_row, :], kernel_size, 0)

# Gather all the results in the root process
if rank == 0:
    blurred_img_full = blurred_img.copy()
    for i in range(1, size):
        start_row = i * rows_per_process
        end_row = start_row + rows_per_process
        blurred_img_full[start_row:end_row, :] = comm.recv(source=i)
else:
    comm.send(blurred_img, dest=0)

# Save the blurred image
if rank == 0:
    cv2.imwrite('blurred.png', blurred_img_full)
