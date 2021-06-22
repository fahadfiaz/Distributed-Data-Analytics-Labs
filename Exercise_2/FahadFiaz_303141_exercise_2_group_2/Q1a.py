from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
Master = 0
t_start = MPI.Wtime()

def NsendAll():
    if rank == Master:
        array_size = 10**7
        array = np.random.randint(1, 10,array_size)
        for process in range(1, size):
            comm.send(array, dest=process)
    else:
        array_recieved = comm.recv(source=Master)  
    comm.barrier()
    return

NsendAll()
if rank == Master:
    print("Total working time: {}".format(MPI.Wtime() - t_start))
