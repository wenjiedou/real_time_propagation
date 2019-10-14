import numpy
from numpy import random
#import scipy
#from scipy.io.numpyio import fwrite

a = random.rand(8,8)
b = random.rand(4,4,4)
b = b.reshape(4,16)

#fd = open('fake-fock.npy','rb')
#fwrite(fd,a.size,a)
#fd.close()
a.tofile('fake-fock.npy')
numpy.savetxt('fake-fock-txt',a)

#fd = open('fake-d.npy','rb')
#fwrite(fd,b.size,b)
#fd.close()
#numpy.save('fake-d',b)
numpy.savetxt('fake-d',b)
