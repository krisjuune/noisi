# -*- coding: utf-8 -*-

'''
wavefield for noisi from axisem3d
'''
from mpi4py import MPI
import h5py
import numpy as np
from pathlib import Path
from netCDF4 import Dataset
# from surface_utils import *
import json
import os
from glob import glob
from noisi_v1.util.geo import geograph_to_geocent
from noisi_v1.util.filter import *
from obspy.signal.interpolation import lanczos_interpolation
try:
    from scipy.signal import sosfilt
except ImportError:
    from obspy.signal._sosfilt import _sosfilt as sosfilt
try:
    from scipy.signal import tukey
except ImportError:
    print("Sorry, no Tukey taper (old scipy version).")
    from obspy.signal.invsim import cosine_taper
from math import ceil, floor
import sys

# input
input_dirs = ["list", "of", "dirs"] # one directory per receiver location
for i in range(len(input_dirs)):
    input_directory = "greens_folder/" + input_dirs[i] + "/output/stations"
    output_directory = "project_name/greens"
    sourcegrid_file = "project_name/sourcegrid.npy"
    f_out_name = 'NET.STA0..MXZ.' + input_dirs[i] + '.gauss.larger.h5'
    process_filter = {'type': 'bandpass',
                        'corners': 4,
                        'zerophase': True,
                        'freq_max': 3.0,
                        'freq_min': 0.1}
    # zero pad the seismograms to avoid edge effects from filtering
    process_pad_factor = 4.
    process_taper = {'type': 'tukey',
                    'alpha': 0.1,
                    'only_trace_end': True} # taper applied to trace end because the seismogram "breaks off" wherever the simulation stops
    process_decimate = {'new_sampling_rate': 40.0,
                        'lanczos_window_width': 25.0} # choosing a sampling rate about 3-4 times the highest target freq. is probably good
    channel = 'MXZ'
    dquantity = 'DIS' #DIS (-placement), VEL (-ocity) or ACC (-celeration). Can be converted so not too important




    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    def get_trace_from_ascii(input_file, input_directory, comp='Z'):
        path = Path(input_directory)
        # file handle
        file = path/input_file 
        # open file and retrieve data
        raw_data = open(file, 'r')
        raw_data = raw_data.read()
        raw_data = np.fromstring(raw_data, dtype = float, sep=' ')
        if comp == 'Z':
            array = raw_data[3::4]
        elif comp == 'T': 
            array = raw_data[2::4]
        elif comp == 'R':
            array = raw_data[1::4]
        return(array)
        # should return a numpy array for the given file and component


    def process_trace(trace, taper, sos, pad_factor, old_t0, 
                    old_sampling_rate, new_sampling_rate, new_npts):

        # taper
        trace *= taper

        # pad
        n = len(trace)
        n_pad = int(pad_factor * n)
        padded_trace = np.zeros(n_pad)
        padded_trace[n_pad // 2: n_pad // 2 + n] = trace

        # filter
        if process_filter['zerophase']:
            firstpass = sosfilt(sos, padded_trace)
            padded_trace = sosfilt(sos, firstpass[:: -1])[:: -1]
        else:
            padded_trace = sosfilt(sos, padded_trace)
        # undo padding
        trace = padded_trace[n_pad // 2: n_pad // 2 + n]
        trace = np.asarray(trace, order='C')
        trace = lanczos_interpolation(trace, old_start=old_t0, 
                                    old_dt=1./old_sampling_rate,
                                    new_start=old_t0, 
                                    new_dt=1./new_sampling_rate,
                                    new_npts=new_npts,
                                    a=process_decimate['lanczos_window_width'],
                                    window='lanczos')
        return(trace)


    # find all the files
    # TODO is this needed?
    files_input = glob(os.path.join(input_directory, ".ascii"))


    # time step and n_time
    def get_time(input_file, input_directory):
        path = Path(input_directory)
        # file = 'II.' + str(station_nr) + '.RTZ.ascii'
        # file handle
        file = path/input_file 
        # Open file and retrieve data
        raw_data = open(file, 'r')
        raw_data = raw_data.read()
        raw_data = np.fromstring(raw_data, dtype = float, sep=' ')
        var_time = raw_data[0::4]
        return(var_time)
    var_time = get_time('II.ST0.RTZ.ascii', input_directory)
    nstep = len(var_time)
    Fs = 1. / (np.mean(var_time[1:] - var_time[:-1]))
    print("old sampling rate: ", Fs)
    solver_dtype = var_time.dtype

    # get config
    comp = channel[-1]
    # read sourcegrid
    f_sources = np.load(sourcegrid_file)
    ntraces = f_sources.shape[-1] 
    new_npts = floor((nstep / Fs) *
                    process_decimate['new_sampling_rate'])

    f_out_name = os.path.join(output_directory, f_out_name)
    if os.path.exists(f_out_name):
        raise ValueError("File %s exists already." % f_out_name)

    comm.barrier()
    if rank == 0:
        os.system('mkdir -p ' + output_directory)
        f_out = h5py.File(f_out_name, "w")

        # DATASET NR 1: STATS
        stats = f_out.create_dataset('stats', data=(0, ))
        stats.attrs['data_quantity'] = dquantity
        stats.attrs['ntraces'] = ntraces
        stats.attrs['Fs'] = process_decimate['new_sampling_rate']
        stats.attrs['fdomain'] = False
        stats.attrs['nt'] = int(new_npts)

        # DATASET NR 2: Source grid
        sources = f_out.create_dataset('sourcegrid', data=f_sources[0:2])

        # DATASET Nr 3: Seismograms itself
        traces_h5 = f_out.create_dataset('data', (ntraces, new_npts),
                                            dtype=np.float32)

    # Define processing
    # half-sided tukey taper
    try:
        taper = tukey(nstep, 0.1)
    except NameError:
        taper = cosine_taper(nstep, 0.1)

    if process_taper['only_trace_end']:
        taper[: nstep // 2] = 1.

    # filter
    if process_filter['type'] == 'lowpass':
        sos = lowpass(freq=process_filter['freq_max'],
                    df=Fs,
                    corners=process_filter['corners'])
    elif process_filter['type'] == 'cheby2_lowpass':
        sos = cheby2_lowpass(freq=process_filter['freq_max'],
                    df=Fs,
                    maxorder=process_filter['max_order'])
    elif process_filter['type'] == 'bandpass':
        sos = bandpass(freqmin=process_filter['freq_min'],
                    freqmax=process_filter['freq_max'],
            corners=process_filter['corners'],df=Fs)
    else:
        raise NotImplementedError('Filter {} is not implemented yet.'.format(
                        process_filter['type']))

    old_t0 = var_time[0]
    old_sampling_rate = Fs
    new_sampling_rate = process_decimate['new_sampling_rate']

    print("Hello, this is rank %g." % rank)
    traces = np.zeros((int(ceil(ntraces / size)), new_npts))
    local_count = 0
    for i in range(ntraces)[rank::size]:
        if i % 1000 == 0: # every 1000th walkthrough
            print('%g / Converted %g of %g traces' % (rank, i, ntraces))
            sys.stdout.flush()
        # read station name, copy to output file
        lat_src = f_sources[1, i]
        lon_src = f_sources[0, i]

        input_file = 'II.ST' + str(i) + '.RTZ.ascii'

        # TODO here:
        # - if you want with the lat, lon values and lat, lon of
        # the source location, you can add the back azimuth 
        # with obspy.geodetics.gps2dist_azimuth, that can be
        # used to rotate ENZ to RTZ and vice versa
        values = get_trace_from_ascii(input_file, input_directory, comp='Z')

        # nothing changes here, just select processing parameters above
        values = process_trace(values, taper, sos, process_pad_factor,
                            old_t0, old_sampling_rate, new_sampling_rate,
                            new_npts)

        if dquantity in ['VEL', 'ACC']:
            new_fs = process_decimate['new_sampling_rate']
            values = np.gradient(values, new_npts * [1. / new_sampling_rate])
            if dquantity == 'ACC':
                values = np.gradient(values, new_npts * [1. / new_sampling_rate])
        # Save in traces array
        traces[local_count, :] = values
        local_count += 1

    # save in temporary npy file
    np.save('traces_%g.npy' % rank, traces[:local_count])

    comm.barrier()

    # rank 0: combine
    if rank == 0:
        #global_count = 0
        for i in range(size):
            traces = np.load('traces_%g.npy' % i)
            traces_h5[i::size,:] = traces
            #global_count += traces.shape[0]
        f_out.close()