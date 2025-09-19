#!/usr/bin/env python
import os
import gc
import time
import h5py
import numpy as np
import torch
import logging
import argparse
from pathlib import Path
from scipy.signal.windows import tukey
from lal import LIGOTimeGPS
import pycbc.psd
from pycbc.types import load_timeseries
from pycbc.waveform import get_fd_waveform
from pycbc.filter import highpass, matched_filter
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta
# import concurrent.futures
# import multiprocessing


class MDCResultTriplet:
    def __init__(self):
        self.time = []
        self.stat = []
        self.var = []

    def add(self, t, s, v):
        self.time.append(float(t))
        self.stat.append(float(s))
        self.var.append(v)

    def dump(self, outfile):
        with h5py.File(outfile, "w") as f:
            f.create_dataset("time", data=np.array(self.time), compression="gzip")
            f.create_dataset("stat", data=np.array(self.stat), compression="gzip")
            f.create_dataset("var", data=np.array(self.var), compression="gzip")


class SignalProcessingParameters:
    def __init__(self, duration, tsegment, fs, low_frequency_cutoff, tfft, tukey_alpha, width_input, height_input):
        # Signal properties
        self.duration = duration
        self.tsegment = tsegment
        self.fs = fs
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = self.fs / 2
        self.dt = 1.0 / self.fs
        self.df = 1.0 / self.duration
        self.tlen = int(self.duration * self.fs)

        # FFT for PSD estimation
        self.tfft = tfft
        self.toverlap = self.tfft / 2
        self.fftlength = int(self.tfft * self.fs)
        self.overlaplength = int(self.fftlength / 2)

        # MF params
        self.tukey_alpha = tukey_alpha
        self.window = tukey(self.tlen, self.tukey_alpha)

        # Image properties
        self.width_input = width_input
        self.height_input = height_input
        # self.kfilter = int(self.fs * self.tsegment / self.width_input)
        # self.kcrop_left = int(self.fs * (self.duration / 2 - 3 * self.tsegment / 4))
        # self.kcrop_right = int(self.fs * (self.duration / 2 + 3 * self.tsegment / 4))
        # self.width_before_smearing = self.kcrop_right - self.kcrop_left


def make_segment_list(duration, t_seg, t_stride=None, toffset_gps=LIGOTimeGPS(0)):
    if t_stride is None:
        t_stride = t_seg
    tstart = -t_stride
    tend = tstart + t_seg
    list_segment = []
    while (True):
        tstart += t_stride
        tend += t_stride
        list_segment.append([tstart + toffset_gps, tend + toffset_gps])
        if tend >= duration:
            break
    return list_segment


def make_psdsegment_list(duration, t_seg, t_stride=None, toffset_gps=LIGOTimeGPS(0)):
    if t_stride is None:
        t_stride = t_seg
    tstart = -t_stride
    tend = tstart + t_seg
    flg_breakloop = False
    list_segment = []
    while (True):
        tstart += t_stride
        tend += t_stride
        if tend >= duration:
            tend = duration
            tstart = tend - t_seg
            flg_breakloop = True
        list_segment.append([tstart + toffset_gps, tend + toffset_gps])
        if flg_breakloop:
            break
    return list_segment


def get_psdindex(mf_segment, psd_segment_list, tbuffer):
    ti, _ = mf_segment
    if ti < psd_segment_list[0][0]:
        raise ValueError("tini is smaller than psd_segment_list[0][0]")
    if psd_segment_list[-1][1] < ti:
        raise ValueError("tini is larger than psd_segment_list[-1][1]")

    for idx, psd_segment in enumerate(psd_segment_list):
        if psd_segment[0] + tbuffer <= ti < psd_segment[1] - tbuffer:
            break
    return idx


def divide_seglist_into_psdsegs(seglist, psdseglist, tbuffer):
    previous_psd_index = -1
    segbuffer = []
    for i in range(len(seglist)):
        psdidx = get_psdindex(seglist[i], psdseglist, tbuffer=tbuffer)
        if psdidx != previous_psd_index:
            segbuffer.append([])
            previous_psd_index = psdidx
        segbuffer[psdidx].append(seglist[i])
    return segbuffer


def get_psdlist(strain, psd_segment_list, sp: SignalProcessingParameters):
    psdlist = []
    avg_method = 'median-mean'
    for psd_segment in psd_segment_list:
        strain_psdseg = strain.crop(psd_segment[0], strain.duration - psd_segment[1])
        psd = strain_psdseg.psd(segment_duration=sp.tfft, avg_method=avg_method)
        psd_interp = pycbc.psd.interpolate(psd, delta_f=1.0 / strain.duration)
        psdlist.append(psd_interp)
    return psdlist


def calculate_matchedfilter(seglist, strain_h1, strain_l1, psdh1_interp, psdl1_interp, template_bank: dict, sp: SignalProcessingParameters):
    '''
    Return: TimeSeries of SNRs
    '''
    # Prepare empty tensor
    nseg = len(seglist)
    snrmap = torch.zeros((nseg, 2, sp.height_input, sp.width_input), dtype=torch.float32)
    # High pass filter
    strain_h1 = highpass(strain_h1, 15.0)
    strain_l1 = highpass(strain_l1, 15.0)
    # Perform template matching
    for i in range(sp.height_input):
        rho_h1 = abs(matched_filter(template_bank['template'][i], strain_h1 * sp.window, psd=psdh1_interp, low_frequency_cutoff=sp.low_frequency_cutoff))
        rho_l1 = abs(matched_filter(template_bank['template'][i], strain_l1 * sp.window, psd=psdl1_interp, low_frequency_cutoff=sp.low_frequency_cutoff))
        for idx, seg in enumerate(seglist):
            snrmap[idx, 0, i] = torch.from_numpy(rho_h1.time_slice(seg[0], seg[1]).numpy())
            snrmap[idx, 1, i] = torch.from_numpy(rho_l1.time_slice(seg[0], seg[1]).numpy())

    # Generate GPS time list
    gpslist = [seg[1] / 2 + seg[0] / 2 for seg in seglist]
    return snrmap, gpslist


# # Copilot proposed
# def process_psd_segment(idxpsd, strain_h1, strain_l1, divided_segment_list, template_bank, sp):
#     # strain_h1 = h1_ts.time_slice(psd_segment_list[idxpsd][0], psd_segment_list[idxpsd][1])
#     # strain_l1 = l1_ts.time_slice(psd_segment_list[idxpsd][0], psd_segment_list[idxpsd][1])
#     psdh1 = strain_h1.psd(segment_duration=sp.tfft, avg_method='median-mean')
#     psdh1_interp = pycbc.psd.interpolate(psdh1, delta_f=1.0 / sp.duration)
#     psdl1 = strain_l1.psd(segment_duration=sp.tfft, avg_method='median-mean')
#     psdl1_interp = pycbc.psd.interpolate(psdl1, delta_f=1.0 / sp.duration)
#     snrmap, gpslist = calculate_matchedfilter(divided_segment_list[idxpsd], strain_h1, strain_l1, psdh1_interp, psdl1_interp, template_bank, sp)
#     # 仮のニューラルネット処理
#     output = torch.empty((len(gpslist),), dtype=torch.float).normal_(0.0, 1.0)
#     # 結果の抽出
#     threshold = 1.0
#     result_triplets = []
#     for i, stat in enumerate(output):
#         if stat >= threshold:
#             result_triplets.append((gpslist[i], stat, 0.5))
#     return result_triplets


def main(args):

    # Time series property
    logging.info('Set the time series property.')
    sp = SignalProcessingParameters(
        duration=16.0,
        tsegment=1.0,
        fs=2048,
        low_frequency_cutoff=20.0,
        tukey_alpha=1.0 / 16.0,
        tfft=4.0,
        width_input=2048,
        height_input=256
    )

    # Create a template bank
    logging.info('Creating a template bank.')
    approximant_tmp = 'IMRPhenomXPHM'
    mcmin_tmp = 5.0
    mcmax_tmp = 50.0
    ngrid_mc = sp.height_input
    mclist = np.logspace(np.log10(mcmin_tmp), np.log10(mcmax_tmp), ngrid_mc, endpoint=True)
    eta = 0.25
    a1 = 0.0
    a2 = 0.0
    template_bank = {'mchirp': mclist, 'template': []}
    template_cache = []
    for i in range(ngrid_mc):
        mass1 = mass1_from_mchirp_eta(mclist[i], eta)
        mass2 = mass2_from_mchirp_eta(mclist[i], eta)
        params_tmp = {
            'approximant': approximant_tmp,
            'mass1': mass1,
            'mass2': mass2,
            'spin1z': a1,
            'spin2z': a2,
            'f_lower': sp.low_frequency_cutoff,
            'delta_f': 1.0 / sp.duration,
            'f_final': sp.high_frequency_cutoff
        }

        hp_fd, _ = get_fd_waveform(**params_tmp)
        template_cache.append(hp_fd)
    template_bank['template'] = template_cache

    # Load a trained neural network
    logging.info('Loading the trained neural network.')
    logging.warning('!!! To be implemented !!!')

    # Get segments
    logging.info('Get segments')
    with h5py.File(args.inputfile, 'r') as file:
        list_start_time = [int(k) for k in file['H1'].keys()]
    list_start_time.sort()

    # Prepare result container
    mdc_results = MDCResultTriplet()

    for start_time in list_start_time:
        # Load strains
        logging.info(f'Start time = {start_time}: Loading strains')
        h1_ts = load_timeseries(args.inputfile, group=f'H1/{start_time}')
        l1_ts = load_timeseries(args.inputfile, group=f'L1/{start_time}')
        duration = h1_ts.duration

        # Make segment list
        tcut = sp.duration / 4
        segment_list = make_segment_list(duration - 2 * tcut, sp.tsegment, sp.tsegment / 2.0, toffset_gps=tcut + h1_ts.start_time)
        # Nsegment = len(segment_list)

        # Make PSD segment list
        psd_segment_list = make_psdsegment_list(duration, sp.duration, sp.duration / 2.0, toffset_gps=h1_ts.start_time)
        divided_segment_list = divide_seglist_into_psdsegs(segment_list, psd_segment_list, tbuffer=tcut)
        assert len(psd_segment_list) == len(divided_segment_list), "PSD segment list and Divided segment list do not have the same length."
        Npsdsegs = len(psd_segment_list)

        # Prepare empty tensors
        logging.info(f'Start time = {start_time}: Making SNR maps')

        tik = time.time()
        for idxpsd in range(Npsdsegs):
            # Crop the strain
            strain_h1 = h1_ts.time_slice(psd_segment_list[idxpsd][0], psd_segment_list[idxpsd][1])
            strain_l1 = l1_ts.time_slice(psd_segment_list[idxpsd][0], psd_segment_list[idxpsd][1])
            # Estimate PSDs
            psdh1 = strain_h1.psd(segment_duration=sp.tfft, avg_method='median-mean')
            psdh1_interp = pycbc.psd.interpolate(psdh1, delta_f=1.0 / sp.duration)
            psdl1 = strain_l1.psd(segment_duration=sp.tfft, avg_method='median-mean')
            psdl1_interp = pycbc.psd.interpolate(psdl1, delta_f=1.0 / sp.duration)
            # Matched filter
            snrmap, gpslist = calculate_matchedfilter(divided_segment_list[idxpsd], strain_h1, strain_l1, psdh1_interp, psdl1_interp, template_bank, sp)
            # snrmap, gpslist = calculate_matchedfilter_parallelized(divided_segment_list[idxpsd], strain_h1, strain_l1, psdh1_interp, psdl1_interp, template_bank, sp)

            # Process by neural network
            logging.info(f'Start time = {start_time}: Processing SNR maps by the neural network.')
            logging.warning('!!! To be implemented !!!')
            output = torch.empty((len(gpslist),), dtype=torch.float).normal_(0.0, 1.0)

            # Get [time, stat, var]
            logging.info(f'Start time = {start_time}: Summarizing into [time, stat, var] triplets.')
            logging.warning('!!! To be implemented !!!')
            threshold = 1.0
            for i, stat in enumerate(output):
                if stat >= threshold:
                    mdc_results.add(gpslist[i], stat, 0.5)

#         logging.info(f'Start time = {start_time}: Slicing strain data')
#         sliced_strain_h1 = [h1_ts.time_slice(psd_segment_list[idxpsd][0], psd_segment_list[idxpsd][1]) for idxpsd in range(Npsdsegs)]
#         sliced_strain_l1 = [l1_ts.time_slice(psd_segment_list[idxpsd][0], psd_segment_list[idxpsd][1]) for idxpsd in range(Npsdsegs)]
# 
#         logging.info(f'Start time = {start_time}: Processing sliced data in parallel')
#         with concurrent.futures.ProcessPoolExecutor() as executor:
#             futures = [
#                 executor.submit(
#                     process_psd_segment, idxpsd, sliced_strain_h1[idxpsd], sliced_strain_l1[idxpsd],
#                     divided_segment_list, template_bank, sp
#                 )
#                 for idxpsd in range(Npsdsegs)
#             ]
#             for future in concurrent.futures.as_completed(futures):
#                 result_triplets = future.result()
#                 for t, s, v in result_triplets:
#                     mdc_results.add(t, s, v)
        tok = time.time()
        break
    logging.info(f'Elapsed time {tok - tik} seconds for {Npsdsegs} psdsegments')
#     gc.collect()

    # Save result triples
    logging.info(f'Saving result triples')
    logging.warning('!!! To be implemented !!!')
    mdc_results.dump(args.outputfile)
#     gc.collect()
    logging.info('Result saved')


if __name__ == '__main__':
    # multiprocessing.set_start_method('spawn')  # ここで呼ぶ
    parser = argparse.ArgumentParser(description='Process MDC data by neural network.')
    parser.add_argument('-i', '--inputfile', type=str, required=True, help='hdf5 file of strain data.')
    parser.add_argument('-o', '--outputfile', type=str, required=True, help='hdf5 file to be output.')
    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)-8s | %(asctime)s | %(message)s',
                        level=log_level, datefmt='%Y-%m-%d %H:%M:%S')

    assert os.path.exists(args.inputfile), f"Input file {args.inputfile} does not exist."
    assert Path(args.inputfile).suffix == '.hdf', f"Input file must be an hdf5 file. (Given {args.inputfile})"
    assert Path(args.outputfile).suffix == '.hdf', f"Output file must be an hdf5 file. (Given {args.outputfile})"

    main(args)
    # # Kludge solution
    # import sys
    # sys.exit(0)
