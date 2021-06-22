from mpi4py import MPI
import numpy as np

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
Master = 0
t_start = MPI.Wtime()

def EsendAll():
    if rank == Master:
        array_size = 10**7
        array = np.random.randint(1, 10,array_size)
        if size>1:
            comm.send(array,dest=1)
        if size>2:
            comm.send(array,dest=2)
    else:
        recvProc=int((rank-1)/2)
        data=comm.recv(source=recvProc)
        destA=2*int(rank)+1
        destB=2*int(rank)+2
        if destA<size:
            comm.send(data,dest=destA)
        if destB<size:
            comm.send(data,dest=destB)
    comm.barrier()
    return

EsendAll()
if rank == Master:
    print("Total working time: {}".format(MPI.Wtime() - t_start))
