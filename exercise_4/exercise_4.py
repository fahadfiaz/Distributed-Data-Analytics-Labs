from mpi4py import MPI
import numpy as np
import glob

import pyplot as plt

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
Master = 0
t_start = MPI.Wtime()
alpha = 10 ** -11
Y = []
X = []
X_Chunk = []
Y_Chunk = []
Parameters = []
Loss = []
Epoch = 0
flag = False


# Pre_processing function to convert data to specific format
def pre_processing(feature_example):
    total_features = np.zeros(482)
    y = 0.0
    processed_features = feature_example.split(" ")
    if (processed_features[0] != ''):
        y = float(processed_features[0])
    for i in range(len(processed_features)):
        split_features = processed_features[i].split(":")
        if (len(split_features) == 2):
            total_features[int(split_features[0])] = int(split_features[1])
    return total_features, y


def SGD(X_Train, Y_Train, alpha):  # function to apply Stochastic Gradient Descent algorithm
    #global Parameters
    if flag == False:  # checking if you are running sgd for 1st epoch else Parameters will get value from broadcasting
        Parameters = np.zeros(len(X_Train[0]))
    for i in range(len(X_Train)):
        my_prediction = np.dot(Parameters.T, X_Train[i])
        error = Y_Train[i] - my_prediction
        Parameters = Parameters + (2 * (alpha * error))
    return Parameters


def RMSE(X, Y, Global_Parameters_Mean):  # function to calculate RMSE
    Predictions = []
    for i in range(len(X)):
        Predictions.append(np.dot(X[i], Global_Parameters_Mean.T))
    Final_Predictions = np.reshape(np.array(Predictions), -1)
    Error = sum((Y - Final_Predictions) ** 2)
    return Error


while Epoch < 5:
    if rank == Master:  # Master Node Read and Divide Data into Chunks
        Total_data = []
        path = "dataset"
        All_Files = glob.glob(path + "/*.txt")
        for name in All_Files:
            File = open(name)
            File_data = File.read().split("\n")
            Total_data.append(File_data)
        np.random.shuffle(Total_data)  # shuffle complete dataset randomly
        for i in range(len(Total_data)):
            for j in range(len(Total_data[i])):
                x, y = pre_processing(Total_data[i][j])
                X.append(x)
                Y.append(y)
        X = np.array(X)  # feature matrix
        Y = np.array(Y)  # prediction vector
        FullDataset = X.shape[0]

        # splitting Dataset in Test/Train format
        X_Train = X[:(FullDataset * 70) // 100, :]
        X_Test = X[(FullDataset * 70) // 100:, :]
        Y_Train = Y[:(FullDataset * 70) // 100]
        Y_Test = Y[(FullDataset * 70) // 100:]

        # partitioning dataset
        data_size = len(X_Train)
        partition = data_size // size
        for process in range(0, size):  # partition data in equal portions
            start = process * partition
            end = (process + 1) * partition
            if process == size - 1 and end != data_size:
                end = data_size
            X_Chunk.append(X_Train[start:end])
            Y_Chunk.append(Y_Train[start:end])

    # scattering data to child processes
    Feature_chunk = np.array(comm.scatter(X_Chunk))
    Prediction_chunk = np.array(comm.scatter(Y_Chunk))

    # child processes applying SGD learning algorithm to learn parameters
    Parameters = SGD(Feature_chunk, Prediction_chunk, alpha)

    # Master node gathering parameters from child processes
    All_Local_Parameters = comm.gather(Parameters, root=Master)

    if (rank == Master):
        if (size > 1):  # checking if there were multiple child processes
            Global_Parameters_Mean = np.mean(All_Local_Parameters,
                                             axis=0)  # Master node calculate Global mean fo parameters
            Loss.append(RMSE(X_Test, Y_Test, Global_Parameters_Mean))  # calculating Loss
        else:
            Global_Parameters_Mean = All_Local_Parameters  # If there was single process
    else:
        Global_Parameters_Mean = None

    Global_Parameters_Mean = comm.bcast(Global_Parameters_Mean, root=Master) # sending average of parameters to child proceses for next epoch

    if len(Global_Parameters_Mean) > 0:
        flag = True
        Parameters = Global_Parameters_Mean
    Epoch = Epoch + 1
    X = []
    Y = []
    X_Chunk = []
    Y_Chunk = []

if (rank == Master):
    print("Final Loss: ", Loss[0])
    print("Total working time: {}".format(MPI.Wtime() - t_start))