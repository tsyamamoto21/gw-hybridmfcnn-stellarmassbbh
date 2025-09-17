#!/usr/bin/env python
import os
import h5py
import numpy as np
import torch
import logging
import argparse
from tqdm import tqdm
from pathlib import Path
from scipy.signal.windows import tukey
import pycbc.psd
from pycbc.types import load_timeseries
from pycbc.waveform import get_fd_waveform
from pycbc.filter import highpass, matched_filter
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta


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

        # Image properties
        self.width_input = width_input
        self.height_input = height_input
        # self.kfilter = int(self.fs * self.tsegment / self.width_input)
        # self.kcrop_left = int(self.fs * (self.duration / 2 - 3 * self.tsegment / 4))
        # self.kcrop_right = int(self.fs * (self.duration / 2 + 3 * self.tsegment / 4))
        # self.width_before_smearing = self.kcrop_right - self.kcrop_left


def make_segment_list(duration, t_seg, t_stride=None, toffset_gps=0):
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


def get_psdindex(mf_segment, psd_segment_list):
    ti, _ = mf_segment
    if ti < psd_segment_list[0][0]:
        raise ValueError("tini is smaller than psd_segment_list[0][0]")
    if psd_segment_list[-1][1] < ti:
        raise ValueError("tini is larger than psd_segment_list[-1][1]")

    for idx, psd_segment in enumerate(psd_segment_list):
        if psd_segment[0] <= ti <= psd_segment[1]:
            break
    return idx


def get_psdlist(strain, psd_segment_list, sp: SignalProcessingParameters):
    psdlist = []
    avg_method = 'median-mean'
    for psd_segment in psd_segment_list:
        strain_psdseg = strain.crop(psd_segment[0], strain.duration - psd_segment[1])
        psd = strain_psdseg.psd(segment_duration=sp.tfft, avg_method=avg_method)
        psd_interp = pycbc.psd.interpolate(psd, delta_f=1.0 / strain.duration)
        psdlist.append(psd_interp)
    return psdlist


def generate_matchedfilter_image(strain_h1, strain_l1, psdh1_interp, psdl1_interp, template_bank: dict, sp: SignalProcessingParameters, flg_lastsegment=False):
    # Devide segment
    seglist = make_segment_list(sp.duration // 2 + sp.tsegment / 2, sp.tsegment, sp.tsegment / 2, sp.duration // 4)
    nseg = len(seglist)
    # SNR image array
    snrlist = torch.zeros((nseg, 2, sp.height_input, sp.width_input), requires_grad=False, dtype=torch.float32)
    # Tukey window
    window = tukey(sp.tlen, sp.tukey_alpha)
    strain_h1 = highpass(strain_h1, 15.0)
    strain_l1 = highpass(strain_l1, 15.0)
    # Perform template matching
    for i in range(sp.height_input):
        rho_h1 = matched_filter(template_bank['template'][i], strain_h1 * window, psd=psdh1_interp, low_frequency_cutoff=sp.low_frequency_cutoff)
        rho_l1 = matched_filter(template_bank['template'][i], strain_l1 * window, psd=psdl1_interp, low_frequency_cutoff=sp.low_frequency_cutoff)
        for j, seg in enumerate(seglist):
            snrlist[j, 0, i] = torch.from_numpy(abs(rho_h1.crop(seg[0], sp.duration - seg[1])).numpy()).to(torch.float32)
            snrlist[j, 1, i] = torch.from_numpy(abs(rho_l1.crop(seg[0], sp.duration - seg[1])).numpy()).to(torch.float32)
    # GPS time array
    gpstimelist = torch.empty((len(seglist), 1), dtype=torch.double)
    for j, seg in enumerate(seglist):
        gpstimelist[j] = float(strain_h1.start_time) + (seg[1] + seg[0]) / 2.0
    return snrlist, gpstimelist


def main(args):

    # Time series property
    logging.info('Set the time series property.')
    sp = SignalProcessingParameters(
        duration=16,
        tsegment=1,
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
        template_bank['template'].append(hp_fd)

    # Load a trained neural network
    logging.info('Loading the trained neural network.')
    logging.warning('!!! To be implemented !!!')

    # Get segments
    logging.info('Get segments')
    with h5py.File(args.inputfile, 'r') as file:
        list_start_time = [int(k) for k in file['H1'].keys()]
    list_start_time.sort()

    for start_time in list_start_time:
        # Load strains
        logging.info(f'Start time = {start_time}: Loading strains')
        h1_ts = load_timeseries(args.inputfile, group=f'H1/{start_time}')
        l1_ts = load_timeseries(args.inputfile, group=f'L1/{start_time}')

        # Estimate PSDs
        duration = h1_ts.duration.item()
        psd_segment_list = make_segment_list(duration, 16.0, 8.0)
        Npsd = len(psd_segment_list)
        snrmap = torch.empty((0, 2, sp.height_input, sp.width_input), dtype=torch.float32)
        gpstensor = torch.empty((0, 1), dtype=torch.float)
        logging.info(f'Start time = {start_time}: Making SNR maps')
        with tqdm(total=Npsd) as pbar:
            for idx_psd in range(Npsd):
                # Progress bar
                pbar.set_postfix({'psd start time': psd_segment_list[idx_psd][0]})
                # Crop the strain
                strain_h1 = h1_ts.crop(psd_segment_list[idx_psd][0], duration - psd_segment_list[idx_psd][1])
                strain_l1 = l1_ts.crop(psd_segment_list[idx_psd][0], duration - psd_segment_list[idx_psd][1])
                # Estimate PSDs
                psdh1 = strain_h1.psd(segment_duration=sp.tfft, avg_method='median-mean')
                psdh1_interp = pycbc.psd.interpolate(psdh1, delta_f=1.0 / sp.duration)
                psdl1 = strain_h1.psd(segment_duration=sp.tfft, avg_method='median-mean')
                psdl1_interp = pycbc.psd.interpolate(psdl1, delta_f=1.0 / sp.duration)
                # Make SNR map input data
                flg_lastsegment = False
                snrmap_seg, gpstime_seg = generate_matchedfilter_image(strain_h1, strain_l1, psdh1_interp, psdl1_interp, template_bank, sp, flg_lastsegment)
                snrmap = torch.cat((snrmap, snrmap_seg))
                gpstensor = torch.cat((gpstensor, gpstime_seg))

                pbar.update()
                if idx_psd == 5:
                    break
        # Process by neural network
        logging.info(f'Start time = {start_time}: Processing SNR maps by the neural network.')
        logging.warning(f'Start time = {start_time}: To be implemented')

        # Get [time, stat, var]
        logging.info(f'Start time = {start_time}: Summarizing into [time, stat, var] triplets.')
        logging.warning(f'Start time = {start_time}: To be implemented')
        break

    # Save result triples
    logging.info(f'Start time = {start_time}: Making SNR maps')
    import pickle
    with open('data/mdc/testoutput.pkl', 'wb') as fo:
        pickle.dump({'snrmap': snrmap, 'gpstime': gpstensor}, fo)


if __name__ == '__main__':
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
