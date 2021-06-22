import numpy as np

v1 = np.random.randint(1,10, size=16, dtype=int)
v2 = np.random.randint(1,10, size=10**2, dtype=int)
print(v1)
# print(v2)
# v3=[v1[2:4],v2[2:4]]
# sum=np.add(v3[0],v3[1])
# print(v3[0])
# print(v3[1])
# print(sum)
print(np.sum(v1[2:4]))