import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ricker, fftconvolve, tukey
from scipy.interpolate import interp1d

# random phase:
p = np.random.rand(1000) * 2. * np.pi
Q = np.zeros(1001, dtype=np.complex)
# the zero'th sample of the spectrum 
# corresponds to the time series average,
# so set to zero. The 1st to last sample
# of the spectrum are used for the random-phase
Q[1: ] = 1. * np.exp(1.j * p)

# just for fun: The spike at 0 is caused by 
# a nonzero mean of the random phase signal
Q_nonzero_mean = Q + 1.
Q_nonzero_mean[0] = 0.0
plt.plot(np.fft.irfft(Q_nonzero_mean))
plt.plot(np.fft.irfft(Q))
plt.xlabel("Time sample nr.")
plt.legend(["Random phase signal with nonzero mean",
           "Random phase signal with zero mean"])
plt.show()

# here is a simple time domain signal "Green fct"
# to use for convolution
s = np.zeros(2001)
s[1500: 1600] = ricker(100, a=20.)
plt.title("Input \"Greens function\"")
plt.plot(s)
plt.show()

# if we perform convolution in freq. domain:
S = np.fft.rfft(s)
plt.plot(np.fft.irfft(Q * S, n=2001))

# and convolution of the same signals,
# but with an algorithm that does it right, 
# starting from time domain signals:
q = np.fft.irfft(Q, n=2001)
plt.plot(fftconvolve(s, q, mode="full"), '--')

# then we see that the signals are identical from the start
# of the causal signal but beforehand, the freq. domain convolved
# signal has wrapped around for periodicity.

# i.e. what we need is zero padding to avoid the wrapping around
S = np.fft.rfft(s, n=2*2001-1)
freq_axis = np.fft.rfftfreq(n=2001)
freq_axis_new = np.linspace(freq_axis.min(), freq_axis.max(), len(S))
# but...the phase cannot be zeropadded without transforming it back to 
# time domain
qint = interp1d(freq_axis, Q, kind="cubic")
Q = qint(freq_axis_new)
td_s = np.fft.irfft(Q * S, n=2*2001-1)

plt.plot(td_s, ':')
plt.legend(["freq. domain, circular convolution", 
            "zero-padding both Green and source",
            "zero-padding only Green"])
plt.show()
