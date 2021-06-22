import numpy as np
import matplotlib.pyplot as plt

A = np.random.random((100, 20))  # Initialize A matrix of dimension 100 × 20 with random values
mu, sigma = 2, 0.01  # mean and standard deviation
v = np.random.normal(mu, sigma, (20,
                                 1))  # This function generates a vector of dimension 20 x 1 with sample of numbers drawn from the normal distribution.
c = np.dot(A,
           v)  # Iterative multiply (element-wise) each row of matrix A with vector v and sum the result of each iteration in another vector c
std = np.std(c)  # Calculate standard deviation of the new vector c
mean = np.mean(c)  # Calculate mean of the new vector c
plt.style.use('fivethirtyeight')  # using pre-defined style provided by Matplotlib.
plt.title("Matrix Multiplication")  # Set title for the axes.
plt.hist(c,bins=5, color='#30475e', edgecolor="black")
plt.show()