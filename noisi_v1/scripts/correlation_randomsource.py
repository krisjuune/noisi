import numpy as np
from noisi_v1.my_classes import WaveField, NoiseSource
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d
from obspy.signal.invsim import cosine_taper
import matplotlib.pyplot as plt
from noisi_v1.util.windows import my_centered


def kristiinas_source_generator(duration_in_samples, n_sources=1, domain="time"):
    if domain == "time":
        # generate random onsets of source
        a = np.zeros(duration_in_samples)
        rix = np.random.randint(0, duration_in_samples, n_sources)
        a[rix] = 1.0

    elif domain == "frequency":
        # generate random phase spectrum
        freq = np.fft.rfftfreq(n=duration_in_samples)
        a = np.random.random(freq.shape)
    else:
        raise ValueError("Unknown domain " + domain)

    return(a)


def generate_timeseries(input_files, all_conf, nsrc,
                        all_ns, td_taper, sourcegrid, debug_plot, 
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
    
    npad = all_ns[1]
    fs = all_ns[3]

    # first the Green's functions are opened
    wf1 = WaveField(input_files[0])
    if input_files[0] != input_files[1]:
        wf2 = WaveField(input_files[1])
    else: # autocorrelation: only one Trace is needed
        wf2 = None

    # the noise source power spectral density model is opened
    nsrc = NoiseSource(nsrc)
    # allocate arrays for the output
    duration_in_samples = int(round(all_conf["timeseries_duration_seconds"] * fs))
    trace1 = np.zeros(duration_in_samples)
    if wf2 is not None:
        trace2 = np.zeros(duration_in_samples)
    else:
        trace2 = None
    
    
    freq_axis = np.fft.rfftfreq(n=duration_in_samples, d=1. / fs)
    fd_taper = cosine_taper(len(freq_axis), 0.02)


    # loop over source locations
    for i in range(wf1.stats["ntraces"]):

        # read green's functions
        g1 = np.ascontiguousarray(wf1.data[i, :] * td_taper)
        if trace2 is None:
            g2 = g1
        else:
            g2 = np.ascontiguousarray(wf2.data[i, :] * td_taper)

        # a) generate a random phase spectrum and add to noise source
        if domain == "frequency":
            # read source power spectral density
            source_amplitude = nsrc.get_spect(i)
            if source_amplitude.sum() == 0:
                continue


            # Question for Eleonore: Should we take the square root of the Power spectrum 
            # and multiply by df?

            # since the spectrum of the noise source model in noisi is differently sampled
            fdfreq = np.fft.rfftfreq(n=npad, d=1. / fs)
            f = interp1d(fdfreq, source_amplitude, kind="cubic")
            source_amplitude = f(freq_axis) * fd_taper
            # plt.plot(freq_axis, source_amplitude, "k")
            # plt.plot(fdfreq, nsrc.get_spect(i), "r--")


            # call the function to get the random phase spectrum
            source_phase = kristiinas_source_generator(duration_in_samples,
                                                       domain="frequency")
            # By definiton: 
            # complex number z = A exp(i phi)
            P = source_amplitude * np.exp(1.j * 2. * np.pi * source_phase)  # phase now between 0 and 2 pi
            #P = np.exp(1.j * 2. * np.pi * source_phase)  # phase now between 0 and 2 pi

            # Convolution of random source and Green's function scaled by surface area of this source location
            p = np.fft.irfft(P, n=duration_in_samples)
            
            # time series 1
            trace1 += fftconvolve(g1, p, mode="full")[0: duration_in_samples] * nsrc.surf_area[i]
            # time series 2
            if trace2 is not None:
                trace2 += fftconvolve(g2, p, mode="full")[0: duration_in_samples] * nsrc.surf_area[i]
            # plot
            if debug_plot and i % 500 == 0:
                print("Created {} of {} source spectra.".format(i, wf1.stats["ntraces"]))
                ts = fftconvolve(g1, p, mode="full")[0: duration_in_samples] * nsrc.surf_area[i]
                plt.plot(ts / ts.max() + i * 0.005, color = 'darkblue')

        # b) define a time series with random "onsets" in time domain, and run convolution in the frequency domain by scipy
        elif domain == "time":
            # fdfreq = np.fft.rfftfreq(n=npad, d=1. / fs)
            # f = interp1d(fdfreq, nsrc.get_spect(i), kind="cubic")
            # source_amplitude = f(freq_axis)
            source_amplitude = np.fft.irfft(nsrc.get_spect(i), n=npad)
            if source_amplitude.sum() == 0:
                continue
            # steps here would be:
            # call the function that gets the random onset time series
            source_phase = kristiinas_source_generator(duration_in_samples,
                                                       domain="time",
                                                       n_sources=n_sources) / np.sqrt(n_sources)
            # it becomes a bit more complicated if there is spatial correlation
            source = fftconvolve(source_amplitude, source_phase, mode="full")[0: duration_in_samples]
            trace1 += fftconvolve(g1, source, mode="full")[0: duration_in_samples] * nsrc.surf_area[i]

            if trace2 is not None:
                trace2 += fftconvolve(g2, source, mode="full")[0: duration_in_samples] * nsrc.surf_area[i]

            if i % 500 == 0:
                print("Created {} of {} source spectra.".format(i, wf1.stats["ntraces"]))
                ts = fftconvolve(g1, source, mode="full")[0: duration_in_samples] * nsrc.surf_area[i]
                plt.plot(ts / ts.max() + i * 0.005, color = 'darkblue')

    
    if debug_plot:
        plt.title("Example signals from individual source locations.")
        plt.xlabel("Sample nr (-)")
        plt.savefig('example_signals.png')
        plt.show()
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

maximum_lag_in_samples = int(round(maximum_lag_in_seconds * fs))
wlen_in_samples = int(round(wlen_in_seconds * fs))
import matplotlib.pyplot as plt

trace1, trace2, source =  generate_timeseries(input_files, all_conf, nsrc,
                                             all_ns, taper, sourcegrid, debug_plot, domain="time",
                                             n_sources=30)
corr = get_correlation(trace1, trace2, wlen_in_samples, maximum_lag_in_samples)


fig = plt.figure(figsize=(8, 6))
duration_in_samples = int(round(all_conf["timeseries_duration_seconds"] * all_ns[-1]))
taxis = np.linspace(0, all_conf["timeseries_duration_seconds"], duration_in_samples)

#if domain == "time":
ax = fig.add_subplot(321)
plt.plot(taxis, source)
plt.xlabel("Time (s)")
plt.ylabel("Ex. src (-)")
ax.set_title("Source comb (time domain)", fontsize="medium")

ax = fig.add_subplot(323)
plt.plot(taxis, trace1, 'g')
plt.xlabel("Time (s)")
plt.ylabel("DIS (m)")
ax.set_title("Example noise trace", fontsize="medium")


ax = fig.add_subplot(325)
lagaxis = np.linspace(-maximum_lag_in_seconds, maximum_lag_in_seconds, len(corr))
plt.plot(lagaxis, corr, "purple")
plt.legend(["Max arr:\n%4.2f s" %(lagaxis[np.argmax(corr)])])
ax.set_title("Stacked cross-corr", fontsize="medium")

plt.xlabel("Lag (s)")
plt.ylabel("C")


trace1, trace2, source =  generate_timeseries(input_files, all_conf, nsrc,
                                             all_ns, taper, sourcegrid, debug_plot, domain="frequency",
                                             n_sources=3600)
corr = get_correlation(trace1, trace2, wlen_in_samples, maximum_lag_in_samples)

ax = fig.add_subplot(322)

freqaxis = np.fft.rfftfreq(n=duration_in_samples, d=1 / fs)
plt.plot(freqaxis, source * np.pi)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Phase (radians)")
plt.yticks([0.0, np.pi / 2., np.pi], ["0.0", "\u03C0", "2 \u03C0"])
ax.set_title("Random phase spectrum", fontsize="medium")



ax = fig.add_subplot(324)
plt.plot(taxis, trace2, 'r')
plt.xlabel("Time (s)")
plt.ylabel("DIS (m)")
ax.set_title("Example noise trace", fontsize="medium")


ax = fig.add_subplot(326)
lagaxis = np.linspace(-maximum_lag_in_seconds, maximum_lag_in_seconds, len(corr))
plt.plot(lagaxis, corr, "purple")
plt.legend(["Max arr:\n%4.2f s" %(lagaxis[np.argmax(corr)])])
ax.set_title("Stacked cross-corr", fontsize="medium")

plt.xlabel("Lag (s)")
plt.ylabel("C")



plt.tight_layout()
plt.savefig("basic_randomphase_and_correlations.png")
plt.show()
