from matplotlib import pyplot as plt

processes=[1,2,3,4,6,8]
plt.plot(processes, [2.69/2.69, 2.69/2.65, 2.695/2.09, 2.69/1.42, 2.69/1.42, 2.69/3.16], label='Cluster Size=4')
plt.plot(processes, [2.75/2.75, 2.75/2.26, 2.75/1.98, 2.75/1.40, 2.75/0.68, 2.75/1.93], label='Cluster Size=8')
plt.plot(processes, [4.33/4.33, 4.33/2.47, 4.33/1.77, 4.33/1.01, 4.33/1.17,4.33/1.58], label='Cluster Sizes=12')
plt.title('SpeedUp')
plt.xlabel('# of Processes')
plt.ylabel('Parallel Efficiency')
plt.legend()
plt.show()