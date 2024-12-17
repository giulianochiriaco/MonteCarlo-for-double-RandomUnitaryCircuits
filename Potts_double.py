import numpy as np
from math import pi,cos,sin,sqrt,exp
import random
from Potts_func_double_hom import *
from more_iter import *
import sys

scriptname,Lx,Ly,p,lA,l,Nstep=sys.argv
Lx=int(Lx)
Ly=int(Ly)
lA=int(lA)
p=float(p)
Nstep=int(Nstep)
folder='/home/gchiriac/DoubleRUC/Montecarlo/Double_circuit/Results'#L'+str(L)

n=3
m=1
Q = n*m+1
d=10

q=24#factorial(Q)
#Nstep = 100000
Ntherm = 10000
Nrep = 5
Ninterval = 1#50

AA = Coupling_matr(Q,p,d)
g1 = gxA1(n,m,Q)
g2 = gxA2(n,m,Q)
contour = boundaryS(Lx,lA,g1,g2)

fileEn = folder + '/Energy_Lx_'+str(Lx)+'_Ly_'+str(Ly)+'_p'+str(p)+'_lA'+str(lA)+'.dat'
filesp = folder + '/Spin_Lx_'+str(Lx)+'_Ly_'+str(Ly)+'_p'+str(p)+'_lA'+str(lA)+'.dat'
filesp2 = folder + '/SpinQ_Lx_'+str(Lx)+'_Ly_'+str(Ly)+'_p'+str(p)+'_lA'+str(lA)+'.dat'
fileConfig = folder + '/Wrong3/Config_Lx_'+str(Lx)+'_Ly_'+str(Ly)+'_p'+str(p)+'_lA'+str(lA)+'.dat'

spin1 = np.array(np.zeros((Ly,Lx,l)),dtype=np.uint8)#3*np.ones((20,10),dtype=np.uint8)#
spin1[-1,:,0] = contour[0]
EBu,ECo,sp,sp2 = Montecarlo(Lx,Ly,spin1,q,AA,Nstep,boundary=contour,Ntherm=Ntherm,prnt=0,Ninterval=Ninterval,config=0)
#EBu is the array of the bulk energies for every sampled configuration, ECo is the energy of the contour, sp is the average spin array, sp2 is the average squared spin array
with open(fileEn,'ab') as f:
    np.savetxt(f,np.transpose(np.array([EBu,ECo])))
with open(filesp,'ab') as f:
    np.savetxt(f,sp)
with open(filesp2,'ab') as f:
    np.savetxt(f,sp2)