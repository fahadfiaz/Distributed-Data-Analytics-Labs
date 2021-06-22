from mpi4py import MPI
import matplotlib.pyplot as plt
import numpy as np
import cv2



comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
Master = 0
t_start = MPI.Wtime()
img=[]
if rank==Master:
    img = cv2.imread('res_2048.jpg')
#    img = cv2.resize(img, (100,100), interpolation = cv2.INTER_AREA)
    rows = img.shape[0]
    partition = rows//size
    partition_rows = []
    for process in range(0, size):
        start = process * partition
        end = (process + 1) * partition
        if process == size - 1 and end != rows:
            end = rows   
        partition_rows.append(img[start:end,:,:])

else:
    partition_rows = None

image_chunk = comm.scatter(partition_rows, root=0)
rows,cols,channels = image_chunk.shape
pixels_value_range=256
chunk_frequency = np.zeros((pixels_value_range,channels),dtype=int)
for row in range(rows):
    for col in range(cols):
        for channel in range(channels):
            for pixel in range(pixels_value_range):
                if image_chunk[row,col,channel] == pixel:
                    chunk_frequency[pixel,channel] = chunk_frequency[pixel,channel] + 1
                
if rank==Master:
    total_image_frequency_count = np.zeros((pixels_value_range,channels), dtype = int)
else:
    total_image_frequency_count = None
    
comm.Reduce([chunk_frequency, MPI.INT],[total_image_frequency_count, MPI.INT],op=MPI.SUM,root=0)

if rank==Master:
    print("Total working time: {}".format(MPI.Wtime() - t_start))
    color = ('b','g','r')
    for channel,col in enumerate(color):
        histr=total_image_frequency_count[:,channel]
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()