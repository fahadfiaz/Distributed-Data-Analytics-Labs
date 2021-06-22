from mpi4py import MPI
import numpy as np
import pandas as pd

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
Master = 0
t_start = MPI.Wtime()
centroid = []
cluster = 12
data = pd.read_csv("Absenteeism_at_work.csv", delimiter=';').values
data_size = data.shape[0]
flag = True

while True:
    if rank == Master:
        if len(centroid) == 0:  # randomly assigning values to centroid from dataset
            random_centroid = np.random.randint(data_size, size=cluster)
            for row in random_centroid:
                centroid.append(data[row])

        partition = data_size // size
        partition_data = []
        for process in range(0, size):  # partition data in equal portions
            start = process * partition
            end = (process + 1) * partition
            if process == size - 1 and end != data_size:
                end = data_size
            partition_data.append(data[start:end])

    else:
        partition_data = None
        updated_centroid = None
        membership = None
        distance_from_centroid = None
        centroid = None

    centroid = comm.bcast(centroid, root=Master)  # broadcasting centroid to slave processes
    received_centroid_size = len(centroid)
    received_partition_data = comm.scatter(partition_data, root=Master)  # scattering chunks of data to slave processes
    received_data_size = received_partition_data.shape[0]

    membership = np.zeros(received_data_size, dtype=int)

    # slave workers calculating membership vector for specific chunk of data
    for i in range(received_data_size):
        distance_from_centroid = []
        for j in range(received_centroid_size):
            dist = (np.sum(centroid[j] - received_partition_data[i])) ** 2  # calculating euclidean distance
            distance_from_centroid.append(np.sqrt(dist))
        membership[i] = np.argmin(distance_from_centroid)  # finding centroid with minimum distance 

    membership_vector = comm.gather(membership, root=Master)  # sending the membership vector to master node

    if rank == Master:  # Master node updating the centroid using membership vector
        membership_vector = [val for sublist in membership_vector for val in sublist]  # flattening list
        updated_centroid = np.zeros((cluster, data.shape[1]), dtype=float)
        count = np.zeros(cluster, dtype=int) 
        for i in range(data_size):
            updated_centroid[membership_vector[i]] += data[i] #add the dataset rows which belong to specific centroid
            count[membership_vector[i]] += 1 #increase count of the specific centroid
        for j in range(cluster):
            if count[j] != 0:  # 
                updated_centroid[j] = updated_centroid[j] / count[j] #mean of centroid
    
    updated_centroid = comm.bcast(updated_centroid, root=0)  # broadcast new centroid to slave processes 
    if (np.array_equal(centroid, updated_centroid)): #check if updated centroid is same as previous then exit
        #print("Total working time {} of process: {}", rank, MPI.Wtime() - t_start)
        flag = False
        break
    else:
        centroid = updated_centroid #update the new centroid
    

if rank == Master and flag == False:
    print("Total working time: {}".format(MPI.Wtime() - t_start))