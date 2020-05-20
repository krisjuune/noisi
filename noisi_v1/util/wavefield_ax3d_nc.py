# %% 
import numpy as np
from netCDF4 import Dataset
import h5py
from glob import glob
from math import floor
from mpi4py import MPI
from scipy.signal import sosfilt, tukey
from obspy.signal.interpolation import lanczos_interpolation
from noisi_v1.util.filter import lowpass
import sys
from obspy.core import Stats, Trace, UTCDateTime


###### specify parameters
runs_list = ['362_148','362_182','364_151','364_179','367_155','367_175','370_158', '370_172', '372_162','372_168','370_158','375_165','378_162','378_168','380_158','380_172','383_155','383_175','386_151','386_179','388_148','388_182']
sourcegrid_file = 'axisem/sourcegrid.npy'
# define for processing 
process_filter = {'type': 'lowpass',
                    'corners': 9,
                    'zerophase': False,
                    'freq_max': 0.2}
process_pad_factor = 10.
process_taper = {'type': 'tukey',
                'alpha': 0.1,
                'only_trace_end': True}
process_decimate = {'new_sampling_rate': 5.0,
                    'lanczos_window_width': 25.0}
channel = 'II' # MXZ for correct
dquantity = 'P' # DIS (-placement), VEL (-ocity), ACC (-eleration), or P (-ressure). Can be converted so not too important
channels = {'Z': 5, 'R': 3, 'T': 4} # ENZ or RTZ, 0 vertical, 1 tranverse, 2 radial
# channels = {'e0': 0, 'e1': 1, 'e2': 2}
prefix = 'axisem/greens/II.bath.RTZ.'
path_to_files = '../../../Desktop/bathy_Greens/'

###### get h5 files

for run in runs_list:

    input_file = path_to_files + run + '/output/stations/axisem3d_synthetics.nc'
    f_out_name = prefix + run + '.h5'
    input_file_path = None

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()


    def process_trace(trace, taper, sos, pad_factor, old_t0, old_sampling_rate,
                    new_sampling_rate, new_npts):
        print(trace)
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

        trace = lanczos_interpolation(trace, old_start=old_t0,
                                    old_dt=1. / old_sampling_rate,
                                    new_start=old_t0,
                                    new_dt=1. / new_sampling_rate,
                                    new_npts=new_npts,
                                    a=process_decimate['lanczos_window_width'],
                                    window='lanczos')
        return(trace)


    # in case of several input files
    if input_file_path is not None:
        input_files = glob(input_file_path + '/*.nc')
        input_file_0 = input_files[0]
    else:
        input_files = []
        input_file_0 = input_file


    # open input file
    nc_syn = Dataset(input_file_0, 'r', format='NETCDF4')
    # time step and n_time, Fs
    var_time = nc_syn.variables['time_points']
    nstep = len(var_time)
    Fs = 1. / (np.mean(var_time[1:] - var_time[:-1]))
    print("old sampling rate: ", Fs)
    solver_dtype = var_time.datatype

    # open source file
    f_sources = np.load(sourcegrid_file)
    # add some metainformation needed
    comp = channel[-1]
    ntraces = f_sources.shape[-1]
    new_npts = floor((nstep / Fs) *
                    process_decimate['new_sampling_rate'])

    # define filter and taper
    taper = tukey(nstep, 0.1)
    if process_taper['only_trace_end']:
        taper[: nstep // 2] = 1.

    # filter
    if process_filter['type'] == 'lowpass':
        sos = lowpass(freq=process_filter['freq_max'],
                    df=Fs,
                    corners=process_filter['corners'])
        # Important ! ! !
        # If using a Chebyshev filter as antialias, that is a good idea, but there
        # is a risk of accidentally cutting away desired parts of the spectrum
        # because rather than the corner frequency, the -96dB frequency is chosen
    else:
        raise NotImplementedError('Filter {} is not implemented yet.'.format(
                                process_filter['type']))

    # initialize output file
    if rank == 0:
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

    comm.barrier()

    # read in traces
    for i in range(ntraces)[rank::size]:
        if i % 1000 == 0:
            print('Getting there slowly, at ' + str(i))
        try:
            if dquantity == 'P':
                # pressure is potential (first output) differentiated twice
                tr = nc_syn.variables['II.ST' + str(i) + '.RTZ.strain'][:, 0]
                tr = Trace(tr).differentiate().differentiate()
                # process traces
                tr = process_trace(tr.data, taper, sos, process_pad_factor,
                                   old_t0=var_time[0],
                                   old_sampling_rate=Fs,
                                   new_sampling_rate=process_decimate[
                                   'new_sampling_rate'],
                                   new_npts=new_npts)

                
            else: 
                tr = nc_syn.variables['II.ST' + str(i) + '.RTZ.strain'][:, channels[comp]]
                # process traces
                tr = process_trace(tr, taper, sos, process_pad_factor,
                                   old_t0=var_time[0],
                                   old_sampling_rate=Fs,
                                   new_sampling_rate=process_decimate[
                                   'new_sampling_rate'],
                                   new_npts=new_npts)

        except KeyError:
            continue
        # save in new output file
        traces_h5[i, :] = tr

    comm.barrier()
    f_out.flush()
    f_out.close()
