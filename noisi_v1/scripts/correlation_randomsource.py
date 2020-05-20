import numpy as np
from noisi_v1 import WaveField, NoiseSource
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt
from obspy import Trace


def kristiinas_source_generator(duration, n_sources=1, domain="time"):
 # ...plus other parameters as needed
    # put the function here that generates a random phase spectrum
    # according to parameters given by the user
    if domain == "time":

        a = np.zeros(duration)
        rix = np.random.randint(0, duration, n_sources)
        a[rix] = 1.0

    elif domain == "frequency":

        freq = np.fft.rfftfreq(n=duration)
        a = np.random.random(freq.shape)
    else:
        raise ValueError("Unknown domain " + domain)

    return(a)


def generate_timeseries(input_files, all_conf, nsrc,
                        all_ns, taper, sourcegrid, debug_plot, 
                        get_correlation=False, noisi_src=True,
                        domain="frequency", n_sources=1):
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
    if noisi_src:
        nsrc = NoiseSource(nsrc)
    # in noisi, different frequency bands have different source distributions.
    # Let's work with one frequency band for the moment. Later on, we'll superpose 
    # a couple of frequency spectra in order to expand the spectra from the 
    # oceanographic model.
        ix_f = 0
    else:
        freq_PSD = 0.2 # for now... 
        ndate, PSD, nfreqs, nlons, nlats, unit = load_amplitude(nsrc)

    # loop over the frequencies and superpose outputs, easy for Eleonore's src

    # allocate arrays for the output
    duration = int(round(all_conf["timeseries_duration_seconds"] * wf1.stats["Fs"]))
    trace1 = np.zeros(duration)
    if wf2 is not None:
        trace2 = np.zeros(duration)
    else:
        trace2 = None

    if domain == "frequency":
        npad = wf1.stats['npad']
        freq_axis = np.fft.rfftfreq(n=duration, d=1./wf1.stats["Fs"])
        tempspec1 = np.zeros(len(freq_axis), dtype=np.complex)
        tempspec2 = np.zeros(len(freq_axis), dtype=np.complex)
        fd_taper = cosine_taper(len(freq_axis), 0.1)

    if get_correlation: 
        correlation = np.zeros(all_ns[2])

    # loop over source locations
    for i in range(wf1.stats["ntraces"]):

        # read green's functions
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
            if noisi_src: 
                source_amplitude = nsrc.distr_basis[i, ix_f] * nsrc.spect_basis[ix_f]
            else: # for eleonore's source
                lat = sourcegrid[0,i]
                lon = sourcegrid[0,i]
                ix_f = find_nearest(nfreqs, freq_PSD) # this step unnecessary if loop over freqs
                source_amplitude = PSD[ix_f, find_nearest(nlons, lon), find_nearest(nlats, lat)]

            # Question for Eleonore: Should we take the square root of the Power spectrum 
            # and multiply by df?

            # since the spectrum of the noise source model in noisi is generally 
            f = interp1d(np.fft.rfftfreq(n=wf1.stats["npad"], d=1./wf1.stats["Fs"]), source_amplitude)
            source_amplitude = f(freq_axis)
            # Fourier transform for real input
            # Before Fourier transform, the time series is zero-padded.
            # this can be used here in order to cover the whole time series
            # duration.            
            spec1 = np.fft.rfft(g1, duration)
            spec2 = np.fft.rfft(g2, duration)

            # call the function to get the random phase spectrum
            source_phase = kristiinas_source_generator(duration, domain="frequency")

            # By definiton: 
            # complex number z = A exp(i phi)
            P = 1. * np.exp(1.j * 2. * np.pi * source_phase)
            
            # then: Convolution of random source and Green's function
            # scaled by surface area of this source location
            p = np.fft.irfft(source_amplitude * P, n=duration)
            trace1 += fftconvolve(g1, p, mode="full")[0: duration] * nsrc.surf_area[i]

            # time series 2
            if trace2 is not None:
                trace2 += fftconvolve(g2, p, mode="full")[0: duration] * nsrc.surf_area[i]
                #tempspec2 += spec2 * p / nsrc.surf_area[i]

            # plot
            if debug_plot and i % 500 == 0:
                print("Created {} of {} source spectra.".format(i, wf1.stats["ntraces"]))
                #ts = np.fft.irfft(fd_taper * p, n=duration)
                ts = fftconvolve(g1, p, mode="full")[0: duration] * nsrc.surf_area[i]
                plt.plot(ts / ts.max() + i * 0.005)

            # correlation TODO
            # if get_correlation: 
            #     c = np.multiply(np.conjugate(tempspec1), tempspec2) 
            #     correlation += my_centered(np.fft.ifftshift(np.fft.irfft(c, all_ns[1])),
            #                            all_ns[2]) * nsrc.surf_area[i]

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

            trace1 += fftconvolve(g1, source, mode="full")[0: len(trace1)] * nsrc.surf_area[i] / n_sources

            if trace2 is not None:
                trace2 += fftconvolve(g2, source, mode="full")[0: len(trace1)] * nsrc.surf_area[i] / n_sources

            if debug_plot and i % 500 == 0:
                print("Created {} of {} source spectra.".format(i, wf1.stats["ntraces"]))
                ts = fftconvolve(g1, source, mode="full") * nsrc.surf_area[i] / n_sources
                plt.plot(ts / ts.max() + i * 0.005)

            # if get_correlation: 
            #     # TODO correlations for time domain
            #     correlation += np.correlate(trace1, trace2, mode="valid") 
            #     # should divide by surf_area?? 

    if debug_plot:
        plt.title("Example signals from individual source locations.")
        plt.xlabel("Sample nr (-)")
        plt.savefig('example_signals.png')
        plt.show()
    # if get_correlation: 
    #     if domain == "frequency":
    #         maxlag = int(200) # all_ns[2] * (1/all_ns[3]) gives 400 # ???? made this up 
    #         lag = np.linspace(-maxlag, maxlag, all_ns[2])
    #         plt.plot(lag, correlation / np.max(np.abs(correlation)))
    #     elif domain == "time":
    #         # some method of calculating correlation in time domain....
    #     plt.plot(lag, correlation / np.max(np.abs(correlation)))
    #     plt.xlabel('Correlation lag (s)')
    #     plt.ylabel('Normalized correlation')
    #     plt.show()
    return(trace1, trace2, source_phase)

def generate_timeseries_nonrandom(input_files, all_conf, nsrc,
                                  all_ns, taper, debug_plot=False):

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

    # allocate arrays for the output
    duration = int(round(all_conf["timeseries_duration_seconds"] * wf1.stats["Fs"]))
    trace1 = np.zeros(duration)
    if wf2 is not None:
        trace2 = np.zeros(duration)
    else:
        trace2 = None

    # loop over source locations
    for i in range(wf1.stats["ntraces"]):

        source_t = np.zeros(duration)  #np.ones(duration)
        source_t[int(duration // 10)] = 1.  # change by Laura because if we want to see exclusively the Green's function
        # we need a "delta function" as source; if not, we will see an integral of the Green's function
        source_amplitude = nsrc.distr_basis[i, ix_f]
        source_t *= source_amplitude
        
        g1 = np.ascontiguousarray(wf1.data[i, :] * taper)
        if trace2 is None:
            g2 = g1
        else:
            g2 = np.ascontiguousarray(wf2.data[i, :] * taper)
        
        trace1 += fftconvolve(g1, source_t, mode="full")[0: len(trace1)] * nsrc.surf_area[i]  
        # added: normalization by surface area

        if trace2 is not None:
            trace2 += fftconvolve(g2, source_t, mode="full")[0: len(trace1)] * nsrc.surf_area[i]

            if debug_plot and i % 300 == 0:
                ts = fftconvolve(g2, source_t, mode="full") * nsrc.surf_area[i]

                plt.plot(ts / ts.max() + i * 0.005)
    if debug_plot:
        plt.title("Example signals from individual sources")
        plt.xlabel("Time sample nr (-)")
        plt.show()
    return(trace1, trace2, source_t)


def get_moveout(input_files, nsrc, all_conf, all_ns, taper,
                debug_plot=False, domain="frequency",
                n_sources=1, j=3, n_loc=[38.7, -15.5]):
    """
    Get moveout plot of convolved traces for every j-th station listed in input_files. 
    """
    n = len(input_files) 
    j = int(j)
    duration = int(round(all_conf["timeseries_duration_seconds"] * all_ns[-1]))
    taxis = np.linspace(0, all_conf["timeseries_duration_seconds"], duration)

    # go through files for sorting
    for i in range(n):
        specifier_i = input_files[i]
        specifier_i = specifier_i[26:33]
        lat = int(specifier_i[:3])/10
        lon = int(specifier_i[-3:])/(-10)
        # initialise list & calculate src-recv distances
        if i == 0: 
            a, b = get_cartesian_distance(lon, lat, src_lat=n_loc[0], src_lon=n_loc[1])
            c = np.sqrt(a**2 + b**2)
            sorted_files = [[input_files[i], c]]
        else: # append outputs for all other recv locations
            a, b = get_cartesian_distance(lon, lat, src_lat=n_loc[0], src_lon=n_loc[1])
            c = np.sqrt(a**2 + b**2)
            sorted_files += [[input_files[i], c]]
    # sort input_files for src-recv distances
    sorted_files = sorted(sorted_files, key = lambda x: x[1])

    fig = plt.figure()
    lwidth = 0.75
    l1 = 'solid'
    c1 = '#1f77b4'
    ticks = np.zeros(int(n/j))
    labels = [' ']*int(n/j)
    # go through sorted files for plotting
    for i in range(n): 
        file_i = sorted_files[i][0]
        # get trace for receiver i
        trace_i, trace2, source = generate_timeseries([file_i, file_i], 
                                                      all_conf, nsrc, all_ns, taper, sourcegrid, 
                                                      debug_plot=debug_plot, domain=domain, 
                                                      n_sources=1)

        # plot signal at every j-th station on moveout plot
        if i == 0:
            plt.plot(taxis, trace_i, linewidth = lwidth, linestyle = l1, color = c1)
            labels[int(i/j)] = file_i[26:33]
            dx = 1.8*max(trace_i)
            axes = plt.gca()
            axes.set_ylim([-1.2*max(trace_i),(int(n/j)-0.3)*dx])
            axes.get_yaxis().set_visible(False)
        elif i % j == 0: 
            plt.plot(taxis, trace_i + int(i/j)*dx, linewidth = lwidth, linestyle = l1, color = c1)
            labels[int(i/j)] = file_i[26:33]
        ticks[int(i/j)] = dx*int(i/j)

    axes.set_yticks(ticks)
    axes.set_yticklabels(labels)
    axes.get_yaxis().set_visible(True)
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('Station')
    plt.show()


def get_correlation(trace1, trace2, wlen_samples, mlag_samples):
    ix = 0
    nw = 0
    while ix < len(trace1) - wlen_samples:
        win1 = trace1[ix: ix + wlen_samples]
        win2 = trace2[ix: ix + wlen_samples]

        wcorr = fftconvolve(win1[::-1], win2, mode="same")

        if "correlation" not in locals():
            correlation = np.zeros(len(wcorr))
        correlation += wcorr
        nw += 1
        ix += wlen_samples
    corr_len = 2 * mlag_samples + 1
    return(my_centered(correlation, corr_len) / nw)
            
# -----------------------------------------------------------------------------
# Below, input values are directly provided right now for experimenting without
# having to invoke noisi. Once the script above works to our satisfaction,
# we'll "plug" it into noisi so that we can use the bookkeeping which noisi 
# provides (organizing input and output files etc)

domain = "frequency" # or "time"
input_files = ["axisem/greens/NET.ST0..MXZ.383_175.gauss.larger.h5", 
               "axisem/greens/NET.ST0..MXZ.367_155.gauss.larger.h5"]
nsrc = "axisem/blob_N/iteration_0/starting_model.h5"
sourcegrid = "axisem/sourcegrid.npy"
all_conf = {"timeseries_duration_seconds": 3600}
with WaveField(input_files[0]) as wf_test:
    all_ns = [wf_test.stats["nt"], wf_test.stats["npad"], 0, wf_test.stats["Fs"]]
    fs = wf_test.stats["Fs"]
taper = cosine_taper(all_ns[0])
debug_plot = True
maximum_lag_in_seconds = 100
wlen_in_seconds = 300   # ideally like 10 times max. lag in seconds

# -----------------------------------------------------------------------------

trace1, trace2, source =  generate_timeseries(input_files, all_conf, nsrc,
                                              all_ns, taper, sourcegrid, debug_plot, domain=domain)



maximum_lag_in_samples = int(round(maximum_lag_in_seconds * fs))
wlen_in_samples = int(round(wlen_in_seconds * fs))
corr = get_correlation(trace1, trace2, wlen_in_samples, maximum_lag_in_samples)
corrtrace = Trace(data=corr)
corrtrace.stats.sampling_rate = fs
corrtrace.write("some_correlation.sac", format="SAC")

import matplotlib.pyplot as plt


fig = plt.figure()
fig.add_subplot(411)
duration = int(round(all_conf["timeseries_duration_seconds"] * all_ns[-1]))
taxis = np.linspace(0, duration, duration)

if domain == "time":
    plt.plot(taxis, source)
    plt.xlabel("Time (s)")
    plt.ylabel("Ex. src (-)")

if domain == "frequency":
    freqaxis = np.fft.rfftfreq(n=duration, d=fs)
    plt.plot(freqaxis, source * np.pi)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Phase (radians)")
    plt.yticks([0.0, np.pi / 2., np.pi], ["0.0", "\u03C0", "2 \u03C0"])

fig.add_subplot(412)
plt.plot(taxis, trace1, 'g')
plt.xlabel("Time (s)")
plt.ylabel("DIS (m)")

fig.add_subplot(413)
plt.plot(taxis, trace2, 'r')
plt.xlabel("Time (s)")
plt.ylabel("DIS (m)")


fig.add_subplot(414)
lagaxis = np.linspace(-maximum_lag_in_seconds, maximum_lag_in_seconds, len(corr))
plt.plot(lagaxis, corr, "purple")
plt.xlabel("Lag (s)")
plt.ylabel("C")

plt.tight_layout()
plt.show()
