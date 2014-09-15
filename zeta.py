'''
Computing Riemann's zeta function on the critical strip. For reference, see:
"An electro-mechanical investigation of the Riemann zeta function in the critical strip" Balth. van der Pol
For computing FFT of continous functions see:
Also see http://stackoverflow.com/questions/24077913/discretized-continuous-fourier-transform-with-numpy
'''

from matplotlib.pyplot import plot, show, xlim, ylim
from numpy import *
from numpy.fft import fft, fftfreq, fftshift

def y(x):
    return exp(x/2) - exp(-x/2) * floor(exp(x))

t0 = -50.0
dt = 1e-5
x = arange(t0, -t0, dt)

f = y(x)

# plot(x, f)
# show()

# Compute Fourier transform
g = fft(f)

# Frequency normalization factor is 2*pi/dt
w = fftfreq(f.size) * 2*pi / dt

# We need to multiply g by a phase factor to get
# a discretisation of the continuous Fourier transform
g *= dt * exp(-1j * w * t0) / sqrt(2*pi)

g_abs2 = real(g * g.conj())

xlim((0., 100.0))
ylim((-1e-8, 1e-3))
plot(w, g_abs2)
show()

# zeros = w[(w > 0.0) & isclose(g_abs2, 0, atol=0.0000003)]
# print len(zeros)
# print zeros[:15]
