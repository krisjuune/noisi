import numpy as np
from noisi_v1 import WaveField, NoiseSource
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt


def kristiinas_source_generator(duration, n_sources=1, domain="time"):
 # ...plus other parameters as needed
    # put the function here that generates a random source
    # according to parameters given by the user
    if domain == "time":

        a = np.zeros(duration)
        rix = np.random.randint(0, duration, n_sources)
        a[rix] = 1.0

    elif domain == "frequency":

        freq = np.fft.rfftfreq(n=duration)
        a = np.random.rand(freq.shape[0])
    else:
        raise ValueError("Unknown domain " + domain)

    return(a)


def generate_timeseries(input_files, all_conf, nsrc,
                        all_ns, taper, debug_plot,
                        domain="time", n_sources=1):
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

    # first the Green's functions are opened
    wf1 = WaveField(input_files[0])
    if input_files[0] != input_files[1]:
        wf2 = WaveField(input_files[1])
    else: # autocorrelation: only one Trace is needed
        wf2 = None

    # the noise source power spectral density model is opened
    nsrc = NoiseSource(nsrc)
    # in noisi, different frequency bands have different source distributions.
    # Let's work with one frequency band for the moment. Later on, we'll superpose 
    # a couple of frequency spectra in order to expand the spectra from the 
    # oceanographic model.
    ix_f = 0

    # allocate arrays for the output
    duration = int(round(all_conf["timeseries_duration_seconds"] * wf1.stats["Fs"]))
    trace1 = np.zeros(duration)
    if wf2 is not None:
        trace2 = np.zeros(duration)
    else:
        trace2 = None

    if domain == "frequency":
        #npad = wf1.stats['npad']
        freq_axis = np.fft.rfftfreq(n=duration, d=1. / wf1.stats["Fs"])
        tempspec1 = np.zeros(len(freq_axis), dtype=np.complex)
        tempspec2 = np.zeros(len(freq_axis), dtype=np.complex)
        fd_taper = cosine_taper(len(freq_axis), 0.2)
    # loop over source locations
    for i in range(wf1.stats["ntraces"]):

        # read gren's functions
        g1 = np.ascontiguousarray(wf1.data[i, :] * taper)
        if trace2 is None:
            g2 = g1
        else:
            g2 = np.ascontiguousarray(wf2.data[i, :] * taper)

        # two possibilities here:
        # a) FFT Green's function, define the random spectrum in
        # Fourier domain, and multiply, then inverse FFT

        if domain == "frequency":
            # read source power spectral density
            source_amplitude = nsrc.distr_basis[i, ix_f] * nsrc.spect_basis[ix_f]

            # Question for Eleonore: Should we take the square root of the Power spectrum 
            # and multiply by df?

            f = interp1d(np.fft.rfftfreq(n=wf1.stats["npad"], d=1./wf1.stats["Fs"]), source_amplitude)
            source_amplitude = f(freq_axis)
            # Fourier transform for real input
            # Before Fourier transform, the time series is zero-padded.
            # this can be used here in order to cover the whole time series
            # duration.            
            #spec1 = np.fft.rfft(g1, n=duration)
            #spec2 = np.fft.rfft(g2, n=duration)

            # call the function to get the random phase spectrum
            source_phase = kristiinas_source_generator(duration, domain="frequency")

            # using complex number definitions:
            # x_complex = a + i b
            # tan phi = b / a
            # r = sqrt(a ** 2 + b ** 2)
            # so then
            # b = a tan phi
            # a = sqrt(r ** 2 / (1 + tan phi))

            # Maybe a lot easier:
            # complex number z = A exp(i phi)
            P = 1. * np.exp(1.j * 2. * np.pi * source_phase)
            
            # then: Convolution of random source and Green's function
            # scaled by surface area of this source location
            p = np.fft.irfft(source_amplitude * P, n=duration)
            trace1 += fftconvolve(g1, p, mode="full")[0: duration] / nsrc.surf_area[i]

            # time series 2
            if trace2 is not None:
                trace2 += fftconvolve(g2, p, mode="full")[0: duration] / nsrc.surf_area[i]
                #tempspec2 += spec2 * p / nsrc.surf_area[i]

            # plot
            if debug_plot and i % 500 == 0:
                print("Created {} of {} source spectra.".format(i, wf1.stats["ntraces"]))
                #ts = np.fft.irfft(fd_taper * p, n=duration)
                ts = fftconvolve(g1, p, mode="full")[0: duration] / nsrc.surf_area[i]
                plt.plot(ts / ts.max() + i * 0.005)


        # b) define a time series with random "onsets" in time domain,
        # and run convolution in the frequency domain by scipy
        # I may have driven my PhD advisor insane with my love for the time domain
        elif domain == "time":
            source_amplitude = np.fft.irfft(nsrc.distr_basis[i, ix_f] * nsrc.spect_basis[ix_f],
                                            n=duration)
            # steps here would be:
            # call the function that gets the random onset time series
            source_phase = kristiinas_source_generator(duration, domain="time", n_sources=n_sources)  
            # it becomes a bit more complicated if there is spatial correlation
            source = fftconvolve(source_amplitude, source_phase, mode="full")

            trace1 += fftconvolve(g1, source, mode="full")[0: len(trace1)] / nsrc.surf_area[i] / n_sources

            if trace2 is not None:
                trace2 += fftconvolve(g2, source, mode="full")[0: len(trace1)] / nsrc.surf_area[i] / n_sources

            if debug_plot and i % 300 == 0:
                print("Created {} of {} source spectra.".format(i, wf1.stats["ntraces"]))
                ts = fftconvolve(g1, source, mode="full") / nsrc.surf_area[i] / n_sources
                plt.plot(ts / ts.max() + i * 0.005)

    if debug_plot:
        plt.title("Example signals from individual source locations.")
        plt.xlabel("Sample nr (-)")
        plt.show()
    return(trace1, trace2, source_phase)


# -----------------------------------------------------------------------------
# Below, input values are directly provided right now for experimenting without
# having to invoke noisi. Once the script above works to our satisfaction,
# we'll "plug" it into noisi so that we can use the bookkeeping which noisi 
# provides (organizing input and output files etc)

domain = "frequency" # or "time"
input_files = ["/home/lermert/Desktop/example/example/greens/G.SSB..MXZ.h5", 
               "/home/lermert/Desktop/example/example/greens/MN.BNI..MXZ.h5"]
nsrc = "/home/lermert/Desktop/example/example/source_1/iteration_0/starting_model.h5"
all_conf = {"timeseries_duration_seconds": 7200}
with WaveField(input_files[0]) as wf_test:
    all_ns = [wf_test.stats["nt"], 0, 0, wf_test.stats["Fs"]]
    fs = wf_test.stats["Fs"]
taper = cosine_taper(all_ns[0])
debug_plot = True
# -----------------------------------------------------------------------------

trace1, trace2, source =  generate_timeseries(input_files, all_conf, nsrc,
                                              all_ns, taper, debug_plot, domain=domain)

import matplotlib.pyplot as plt

fig = plt.figure()
fig.add_subplot(311)
duration = int(round(all_conf["timeseries_duration_seconds"] * all_ns[-1]))
taxis = np.linspace(0, all_conf["timeseries_duration_seconds"], duration)

if domain == "time":
    plt.plot(taxis, source)
    plt.xlabel("Time (s)")
    plt.ylabel("An example source \"trigger\" (-)")

if domain == "frequency":
    freqaxis = np.fft.rfftfreq(n=duration, d=fs)
    plt.plot(freqaxis, source * np.pi)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.yticks([0.0, np.pi / 2., np.pi], ["0.0", "\u03C0", "2 \u03C0"])

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
