from numpy import *
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt


dt = 0.0001
t = arange(0, 1, dt)
x1 = sin(600*2*pi*t)
x2 = sin(600*2*pi*t)
x3 = sin(600*2*pi*t) * (1 + sin(10*2*pi*t))

x = x3

plt.figure()

# signal
plt.subplot(4, 1, 1)
plt.plot(t, x)


# squared signal
plt.subplot(4, 1, 2)
x_sqr = x**2
plt.plot(t, x_sqr)


# filter
plt.subplot(4, 1, 3)

nyquist = 1./(2 * dt)
cutoff = 100
[b, a] = butter(N=1, Wn=cutoff/nyquist, btype='low')
x_filt = filtfilt(b, a, x_sqr)

plt.plot(t, x_filt)


plt.show()