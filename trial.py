import numpy as np
from itertools import * 
from more_iter import *
import random


def Coupling_matr(Q,p,d,tol=1e-16):
    "Constructs coupling matrix in the limit of large d for arbitrary Q and p"
    q=factorial(Q)
    Perm_m = np.array([[perm_mult(perm_inv(nth_permutation(range(Q),Q,j)),nth_permutation(range(Q),Q,i)) for j in range(q)] for i in range(q)])
    AA=np.zeros((q,q))
    for i in range(q):
        for j in range(q):
            if np.count_nonzero(Perm_m[i,j]-np.arange(Q)!=0)==2:
                AA[i,j]+=1
    return max(p,tol)+np.identity(q)*(1-p)+(1-p)*AA/d

def U_local(l,Jp1,Jp2,J0):
    q = np.shape(Jp1)[0]
    U = np.zeros((l,l,q,q))
    lv = l-3
    for j in range(l-2,l):
        U[0,j,:,:] = Jp1
        U[j,0,:,:] = Jp1
        U[lv,j,:,:] = Jp2
        U[j,lv,:,:] = Jp2
    for j in range(1,l-3):
        U[j,j+1,:,:] = J0
        U[j+1,j,:,:] = J0
    return U

Jp1 = Coupling_matr(4,0.1,10)#np.array([[1,0,0.2],[0,1,0],[0.2,0,1]])
Jp2 = Coupling_matr(4,0.2,10)#np.array([[1,0,0],[0,1,0.3],[0,0.3,1]])
J0 = Coupling_matr(4,0,10)#np.array([[1,0,0],[0,1,0],[0,0,1]])

UU = U_local(6,Jp1,Jp2,J0)
print(UU[0,5])
s1 = np.array([0,3,0,1,1,2,10,12],dtype=int)

def slice1(UU,s1):
    size = np.shape(s1)[0]
    i, j = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    s1_i = s1[i]  # Shape (6, 6), corresponds to s1[i]
    s1_j = s1[j]  # Shape (6, 6), corresponds to s1[j]
    return UU[i, j, s1_i, s1_j]

def slice2(UU,s1):
    size = np.shape(s1)[0]
    result2=np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            result2[i,j]=UU[i,j,s1[i],s1[j]]
    return result2

import time

"""t0=time.time()
for i in range(100000):
    aac = slice1(UU,s1)
print(time.time()-t0)"""

"""t0=time.time()
for i in range(100000):
    aac = slice2(UU,s1)
print(time.time()-t0)"""

s1= np.random.randint(0, 24, size=(20,10,6))
s1 = np.random.randint(0, 24, size=(20,10,6, 2))

s1_a = s1[..., 0]  # Shape (Ly, Lx, l)
s1_b = s1[..., 1]  # Shape (Ly, Lx, l)
print(np.shape(s1_a[:,:,np.newaxis]))

# Use advanced indexing to construct the output array
#Out = UU[s1_a[:, :, :,np.newaxis], s1_b[:, :, :,np.newaxis], s1_a, s1_b]


#Out = UU[s1[:, :, np.newaxis], s1[:, :, np.newaxis], s1, s1]
#print(np.shape(s1_a))
#Arr = UU[ranA[:,:,:],ranA[:,:,:]]

l = 4
q = 3
Ly = 5
Lx = 6

# Create example arrays
U = np.random.rand(l, l, q, q)  # Random array of shape (l, l, q, q)
s1 = np.random.randint(0, q, size=(Ly, Lx, l))  # Random indices in range [0, q-1]

# Extract indices from s1
s1_a = s1[..., 0]  # Shape (Ly, Lx, l)
s1_b = s1[..., 1]  # Shape (Ly, Lx, l)

a_indices, b_indices = np.indices((l, l))

# Use advanced indexing to construct the output array
Out = np.einsum('...ab, ...cd -> ...abcd', U, s1)# Alternatively, if you want to use np.einsum for some reason
# Note: This might not be necessary, but can be used for similar operations
# Out = np.einsum('...ab, ...cd -> ...abcd', U, s1)

print("Output shape:", Out.shape)
