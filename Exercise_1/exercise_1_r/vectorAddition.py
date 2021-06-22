from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("My rank is : ", rank)
Master = 0
t_start = MPI.Wtime()

if rank == Master:
    vector_size = 10**7
    v1 = np.random.randint(1,10, vector_size, dtype=int)
    v2 = np.random.randint(1,10, vector_size, dtype=int)
    partition = vector_size  // size
    print("v1 is: {}".format(v1))
    print("v2 is: {}".format(v2))
    print("\nNumber of processes: {}".format(size))
    print("Portion Size: {}\n".format(partition))
    for process in range(1, size):
        start = process * partition
        end = (process + 1) * partition

        if process == size - 1 and end != vector_size:
            end = vector_size
		
        print("sent portion 1 is {}".format(v1[start:end]))
        print("sent portion 2 is {}".format(v2[start:end]))
        comm.send([v1[start:end], v2[start:end]], process)

    summed_vector_result = []
    start = Master * partition
    end = (Master + 1) * partition
    print("\nmaster node portion of v1: {} ".format(v1[start:end]))
    print("master node portion of v2: {}".format(v2[start:end]))
    sum_first_portion = np.add(v1[start:end], v2[start:end])
    
    summed_vector_result.append(sum_first_portion)
    for i in range(1, size):
        summed_vector_portions = comm.recv(source=i)
        summed_vector_result.append(summed_vector_portions)
    print("\nFinal summed vector: {}".format(np.array(summed_vector_result).flatten()))
    print("Total working time: {}".format(MPI.Wtime() - t_start))
    
if rank != Master:
    vectors_array_received = comm.recv(source=Master)
    sum_vector_portion = np.add(vectors_array_received[0], vectors_array_received[1])
    comm.send(sum_vector_portion, dest=Master)
    print("Values recv from v1: {}".format(vectors_array_received[0]))
    print("Values recv from v1: {}".format(vectors_array_received[1]))
    print("Values send to master process: {} by process: {}\n".format(sum_vector_portion, rank))
