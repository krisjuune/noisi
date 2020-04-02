import numpy as np
from noisi_v1 import WaveField, NoiseSource
from scipy.signal import fftconvolve
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt


def kristiinas_source_generator(duration, domain="time"): # ...plus other parameters as needed
    # put the function here that generates a random source
    # according to parameters given by the user
    # right now just gives back one single source impulse
    # not very interesting
    # note: this is time-domain but a frequency domain function
    # is also possible
    if domain == "time":
        a = np.zeros(duration)
        a[int(duration // 2)] = 1.0
    else:
        raise ValueError("Unknown domain " + domain)
    return(a)

def generate_timeseries(input_files, all_conf, nsrc,
                        all_ns, taper, debug_plot):
    """
    Generate a long time series of noise at two stations
    (one station in case of autocorrelation)

    input_files: list of 2 wave field files
    all_conf: Configuration object
    nsrc: Path to noise source file containing spatial
    distribution of noise source pressure PSD
    all_ns: Numbers specifying trace lengths / FFT parameters:
    all_ns = (nt, n, n_corr, Fs) where nt is the length
    of the Green's function seismograms in samples,
    n is the length to which it is padded for FFT, 
    n_corr is the nr of samples of the resulting cross-correlation,
    and fs is the sampling rate.
    taper: cosine taper with length nt 
    (tapering helps avoid FFT artefacts)

    """

    wf1 = WaveField(input_files[0])
    if input_files[0] != input_files[1]:
        wf2 = WaveField(input_files[1])
    else: # autocorrelation: only one Trace is needed
        wf2 = None

    nsrc = NoiseSource(nsrc)
    ix_f = 0  # in noisi, different frequency bands have different source distributions.
    # Let's work with one frequency band for the moment and then find the best way to 
    # represent Eleonore's spectra in a simple way in the nsrc file.

    # allocate arrays for the output
    duration = int(round(all_conf["timeseries_duration_seconds"] * wf1.stats["Fs"]))
    trace1 = np.zeros(duration)
    if wf2 is not None:
        trace2 = np.zeros(duration)
    else:
        trace2 = None

    # loop over source locations
    for i in range(wf1.stats["ntraces"]):

        # two possibilities here:
        # a) FFT Green's function, resample, define the random spectrum in
        # Fourier domain, and multiply, then inverse FFT

        # FFT can be done using rfft (real FFT)

        # call the function to get the random phase spectrum

        # For extending the duration of the time series to the desired duration,
        # it would be necessary to increase the spectrum sampling before the inverse FFT.
        # e.g. this could be done by interpolating on a finer frequency array with
        # scipy.interp1d. Not sure how mathematically clean this approach is; please ask Tarje for a comment
        # one foreseeable problem might be that the resulting ifft might contain a nonzero imaginary part, 
        # which we don't want

        # Finally, irfft (inverse fft for real-valued time series) needs to be called to get the time series

        


        # b) define a time series with random "onsets" in time domain,
        # and run convolution in the frequency domain by scipy
        # I may have driven my PhD advisor insane with my love for the time domain
        
        # steps here would be:
        # call the function that gets the random onset time series
        source_t = kristiinas_source_generator(duration, domain="time")  # random should be easy to implement here; 
        # it becomes a bit more complicated if there is spatial correlation
        g1 = np.ascontiguousarray(wf1.data[i, :] * taper)
        if trace2 is None:
            g2 = g1
        else:
            g2 = np.ascontiguousarray(wf1.data[i, :] * taper)
        source_amplitude = nsrc.distr_basis[i, ix_f]

        trace1 += fftconvolve(g1, source_amplitude * source_t, mode="full")[0: len(trace1)]

        if trace2 is not None:
            trace2 += fftconvolve(g2, source_amplitude * source_t, mode="full")[0: len(trace1)]

            if debug_plot and i % 300 == 0:
                ts = fftconvolve(g2, source_amplitude * source_t, mode="full")
                plt.plot(ts / ts.max() + i * 0.005)
    if debug_plot:
        plt.title("Example signals from individual sources.")
        plt.xlabel("Time sample nr (-)")
        plt.show()
    return(trace1, trace2, source_t)


# -----------------------------------------------------------------------------
# Below, input values are directly provided right now for experimenting without
# having to invoke noisi. Once the script above works to our satisfaction,
# we'll "plug" it into noisi so that we can use the bookkeeping which noisi 
# provides (organizing input and output files etc)


input_files = ["/home/lermert/Desktop/example/example/greens/G.SSB..MXZ.h5", 
               "/home/lermert/Desktop/example/example/greens/MN.BNI..MXZ.h5"]
nsrc = "/home/lermert/Desktop/example/example/source_1/iteration_0/starting_model.h5"
all_conf = {"timeseries_duration_seconds": 7200}
with WaveField(input_files[0]) as wf_test:
    all_ns = [wf_test.stats["nt"], 0, 0, wf_test.stats["Fs"]]
taper = cosine_taper(all_ns[0])
debug_plot = True

trace1, trace2, source =  generate_timeseries(input_files, all_conf, nsrc,
                                              all_ns, taper, debug_plot)

import matplotlib.pyplot as plt
fig = plt.figure()
fig.add_subplot(311)
duration = int(round(all_conf["timeseries_duration_seconds"] * all_ns[-1]))
taxis = np.linspace(0, all_conf["timeseries_duration_seconds"], duration)
plt.plot(taxis, source)
plt.xlabel("Time (s)")
plt.ylabel("Source \"trigger\" (-)")

fig.add_subplot(312)
plt.plot(taxis, trace1, 'g')
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")

fig.add_subplot(313)
plt.plot(taxis, trace2, 'r')
plt.xlabel("Time (s)")
plt.ylabel("Displacement (m)")
plt.tight_layout()
plt.show()