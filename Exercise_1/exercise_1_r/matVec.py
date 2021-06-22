from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
print("My rank is : ", rank)
Master = 0
t_start = MPI.Wtime()

if rank == Master:
    m_size = 10**2
    m = np.random.randint(1, 10, size=[m_size, m_size])
    v = np.random.randint(1, 10, size=[m_size, 1])
    partition = m_size // size
    print("m is: {}".format(m))
    print("v is: {}".format(v))
    print("\nNumber of processes: {}".format(size))
    print("Portion Size: {}\n".format(partition))
    for process in range(1, size):
        start = process * partition
        end = (process + 1) * partition
        if process == size - 1 and end != m_size:
            end = m_size
        comm.send(m[start:end, :], dest=process)
        comm.send(v, dest=process)

    start = Master * partition
    end = (Master + 1) * partition
    final_res=np.matmul(m[start:end, :], v)
    for processes in range(1, size):
        final_res=np.append(final_res,comm.recv(source=processes))

    print("\nFinal vector: {}".format(final_res.reshape(m_size,1)))
    print("Total working time: {}".format(MPI.Wtime() - t_start))

if rank != Master:
    matrixChunk = comm.recv(source=Master)
    vector = comm.recv(source=Master)
    product = np.matmul(matrixChunk, vector)
    comm.send(product, dest=Master)