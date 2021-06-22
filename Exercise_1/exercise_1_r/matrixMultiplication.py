from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("My rank is : ", rank)
Master = 0
t_start = MPI.Wtime()

if rank == Master:
    mat_size = 4
    m1_share = np.random.randint(10, size=[mat_size, mat_size], dtype=int)
    m2_share = np.random.randint(10, size=[mat_size, mat_size], dtype=int)
else:
    mat_size = 0
    m1_share = None
    m2_share = None

mat_size = comm.bcast(mat_size, root=Master)
m1_share = comm.bcast(m1_share, root=Master)
m2_share = comm.bcast(m2_share, root=Master)
print(m1_share)
print(m2_share)

if rank == Master:
    partition = mat_size // size
    start = Master * partition
    end = (Master + 1) * partition
    Final_res = np.dot(m1_share[start:end, :], m2_share)

    for processes in range(1, size):
        Final_res = np.append(Final_res, comm.recv(source=processes))

    print("Matrix-matrix multiplication result: {}".format(Final_res.reshape(mat_size, mat_size)))
    print("Total working time: {}".format(MPI.Wtime() - t_start))

if rank != Master:
    partition = mat_size // size
    start = rank * partition
    end = (rank + 1) * partition
    multiplication_chunk_res = np.dot(m1_share[start:end, :], m2_share)
    comm.send(multiplication_chunk_res, dest=Master)
