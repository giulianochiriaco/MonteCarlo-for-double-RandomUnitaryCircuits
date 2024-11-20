import numpy as np
import time
Ly = 40
Lx = 60
l = 8
l2=1
q = 24

# Create an example array s with random integers in the range [0, q-1]
s = np.random.randint(0, q, size=(Ly, Lx, l,l2))

# Create the output array s' with the desired shape
s_prime = np.zeros((Ly, Lx, l, l2,q), dtype=int)

# Use broadcasting to fill s'
# np.arange(q) creates an array [0, 1, 2, ..., q-1]
# We compare s with this array to create the desired output
t0=time.time()
for i in range(100):
    s_prime[np.arange(Ly)[:, np.newaxis, np.newaxis, np.newaxis], 
            np.arange(Lx)[np.newaxis, :, np.newaxis, np.newaxis], 
            np.arange(l)[np.newaxis, np.newaxis, :, np.newaxis],
            np.arange(l2)[np.newaxis, np.newaxis, np.newaxis,:], 
            s] = 1
print(time.time()-t0)

t0=time.time()
for i in range(100):
    np.roll(s,shift=1,axis=1)
print(time.time()-t0)

t0=time.time()
for i in range(100):
    np.roll(s_prime,shift=1,axis=1)
print(time.time()-t0)
