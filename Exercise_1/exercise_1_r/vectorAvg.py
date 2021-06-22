from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("My rank is : ", rank)
Master = 0
t_start = MPI.Wtime()

if rank == Master:
    vector_size = 16
    v1 = np.random.randint(1, 10, vector_size, dtype=int)
    partition = vector_size // size
    print("v1 is: {}".format(v1))
    print("\nNumber of processes: {}".format(size))
    print("Portion Size: {}\n".format(partition))
    for process in range(1, size):
        start = process * partition
        end = (process + 1) * partition
        if process == size - 1 and end != vector_size:
            end = vector_size
        print("sent portion 1 is {}".format(v1[start:end]))
        comm.send(v1[start:end], process)

    start = Master * partition
    end = (Master + 1) * partition
    print("\nmaster node portion of v1: {} ".format(v1[start:end]))
    sum_total_vector = np.sum(v1[start:end])

    for i in range(1, size):
        summed_vector_portion = comm.recv(source=i)
        sum_total_vector += summed_vector_portion
    print("\nAvg of vector: {}".format(sum_total_vector/vector_size ))
    print("Total working time: {}".format(MPI.Wtime() - t_start))

if rank != Master:
    vectors_array_received = comm.recv(source=Master)
    sum_vector_portion = np.sum(vectors_array_received)
    comm.send(sum_vector_portion, dest=Master)
    print("Values recv from v1: {}".format(vectors_array_received))
    print("Values send to master process: {} by process: {}\n".format(sum_vector_portion, rank))
