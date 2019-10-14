import numpy as np
from numpy import linalg as LA
import matplotlib
import matplotlib.pyplot as plt
from scipy import special

Sigma_t_cut = 1000
time_step = 0.05
nbf = 2

Iden = np.identity(nbf)
S = Iden.copy()


SigmaR = np.fromfile('sigma.npy')
F = np.fromfile('F.npy')

SigmaR = SigmaR.reshape((Sigma_t_cut, nbf, nbf, 2))
F = F.reshape((nbf, nbf))
SigmaR = SigmaR[:,:,:,0] + 1j*SigmaR[:,:,:,1]

#==> remove negativity
time_grid = time_step*np.arange(Sigma_t_cut)
for it in range(Sigma_t_cut):
    SigmaR[it,:,:] *= special.erfc(1.0*time_grid[it]/100.0)

TrSigmaR=-1.0*np.einsum('ijk,kj->i',SigmaR,S)


plt.plot(time_grid,TrSigmaR.imag)
plt.plot(time_grid,TrSigmaR.real)
plt.show()


AddTime = 200000
AddSigmaR = np.zeros(AddTime,dtype='complex128')

TotalTime = AddTime + Sigma_t_cut

freq_grid   =   np.fft.fftfreq(TotalTime)*2*np.pi/time_step

SigmaR_iw = np.zeros((TotalTime,nbf,nbf),dtype='complex128')


for i in range(nbf):
    for j in range(nbf):
        TotalSigma = np.append(SigmaR[:,i,j],AddSigmaR)
        SigmaR_iw[:,i,j]   =   np.fft.ifft(TotalSigma)

SigmaR_iw *= time_step*TotalTime

#==> plot
TrSigmaR_iw = np.einsum('iab,ba->i', SigmaR_iw, S, optimize=True)

w, TrSigmaR_iw  =   zip(*sorted(zip(freq_grid, TrSigmaR_iw)))
w    =   np.asarray(w)
TrSigmaR_iw    =   np.asarray(TrSigmaR_iw)

np.savetxt('TrSigmaR_iwim.out', TrSigmaR_iw.imag)
np.savetxt('TrSigmaR_iwre.out', TrSigmaR_iw.real)
#========================================================================
# ==> Spectral 
#========================================================================

TrA_not = np.zeros(TotalTime)
TrA = np.zeros(TotalTime)

for n in range(freq_grid.shape[0]):

    denom = (freq_grid[n]+1j*0.01)*S - F
    one_Gnot_iw = np.linalg.inv(denom)
    denom -= SigmaR_iw[n,:,:]
    one_G_iw = np.linalg.inv(denom)

    TrA_not[n] = -2.0* np.einsum('ab,ba->', np.imag(one_Gnot_iw), S, optimize=True)
    TrA[n] = -2.0* np.einsum('ab,ba->', np.imag(one_G_iw), S, optimize=True)


#==> plot
w, Aw_not, Aw =   zip(*sorted(zip(freq_grid, TrA_not, TrA)))

w    =   np.asarray(w)
Aw_not    =   np.asarray(Aw_not)
Aw    =   np.asarray(Aw)

np.savetxt('w.out', w)
np.savetxt('Aw0.out', Aw_not)
np.savetxt('Aw.out', Aw)

plt.plot(w,Aw_not)
plt.plot(w,Aw)
plt.show()


