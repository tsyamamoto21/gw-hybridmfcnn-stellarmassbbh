#!/usr/bin/env python
import os
import h5py
import torch
import argparse
import numpy as np
from scipy.signal.windows import tukey
from pycbc.types import load_timeseries
import pycbc.psd
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta
from pycbc.waveform import get_fd_waveform
from pycbc.filter import matched_filter
from dl4longcbc.utils import if_not_exist_makedir
import concurrent.futures


def make_snrmap_coarse(snrmap, kfilter):
    nc, nx, ny = snrmap.shape
    ny_coarse = ny // kfilter
    snrmap_coarse = torch.zeros((nc, nx, ny_coarse), dtype=torch.float32)
    for i in range(ny_coarse):
        snrmap_coarse[:, :, i] = torch.sqrt(torch.mean(snrmap[:, :, i * kfilter: (i + 1) * kfilter]**2, dim=-1))
    return snrmap_coarse


class SignalProcessingParameters:
    def __init__(self, duration, tsegment, fs, low_frequency_cutoff, tfft, tukey_alpha, width_input, height_input):
        # Signal properties
        self.duration = duration
        self.tsegment = tsegment
        self.fs = fs
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = self.fs / 2

        # FFT for PSD estimation
        self.tfft = tfft
        self.toverlap = self.tfft / 2
        self.fftlength = int(self.tfft * self.fs)
        self.overlaplength = int(self.fftlength / 2)

        # MF params
        self.mfdatalength = int(self.duration * self.fs)
        self.tukey_alpha = tukey_alpha

        # Image properties
        self.width_input = width_input
        self.height_input = height_input
        self.kfilter = int(self.fs * self.tsegment / self.width_input)
        self.kcrop_left = int(self.fs * (self.duration / 2 - 3 * self.tsegment / 4))
        self.kcrop_right = int(self.fs * (self.duration / 2 + 3 * self.tsegment / 4))
        self.width_before_smearing = self.kcrop_right - self.kcrop_left


class SegmentTimestamps:
    def __init__(self):
        self.start_time_str: str = None
        self.start_time: int = None
        self.end_time: int = None
        self.tclist_for_short_segment: list = None


def get_timestamps(file_foreground: str, file_injection: str, sp: SignalProcessingParameters, nstart: int, nend: int):
    # Get start time and end time
    with h5py.File(file_foreground, 'r') as fo:
        x = fo['H1'].keys()
        start_time_strlist = [xsample for xsample in x]
        start_time_list = [int(s) for s in start_time_strlist]
        nsegment = len(start_time_list)
        end_time_list = [int(len(fo['H1'][start_time_strlist[i]]) / sp.fs) + start_time_list[i] for i in range(nsegment)]

    # Get the injection time stamps
    tclist = [[] for _ in range(nsegment)]
    if file_injection is None:
        # Assign a segment to an injection
        for n in range(nsegment):
            duration = end_time_list[n] - start_time_list[n]
            ntc = int(np.floor(duration / 32))
            for i in range(ntc):
                tclist[n].append(start_time_list[n] + 16 + i * 32)
    else:
        with h5py.File(file_injection, 'r') as fo:
            tclist_from_hdf5 = fo['tc'][:]
        # Assign a segment to an injection
        for idx in range(len(tclist_from_hdf5)):
            for n in range(nsegment):
                if (start_time_list[n] <= tclist_from_hdf5[idx]) * (tclist_from_hdf5[idx] <= end_time_list[n]):
                    tclist[n].append(tclist_from_hdf5[idx])

    # Store timestamps in instances
    segment_timestamp_list = []
    for n in range(nstart, nend):
        st = SegmentTimestamps()
        st.start_time_str = start_time_strlist[n]
        st.start_time = start_time_list[n]
        st.end_time = end_time_list[n]
        st.tclist_for_short_segment = tclist[n]
        segment_timestamp_list.append(st)

    return segment_timestamp_list


def matchedfilter_core(file_foreground: str, template_bank: list, outdir: str, sp: SignalProcessingParameters, timestamps: SegmentTimestamps, offevent=False):

    print(f"[PID {os.getpid()}] Processing {timestamps.start_time}: Binded to {os.sched_getaffinity(0)}")
    # SNR array
    snrlist = torch.zeros((2, sp.height_input, sp.width_before_smearing), requires_grad=False)

    # Tukey window
    window = tukey(sp.mfdatalength, sp.tukey_alpha)

    tclist_for_short_segment = np.array(timestamps.tclist_for_short_segment)
    if offevent:
        tclist_for_short_segment = (tclist_for_short_segment[:-1] + tclist_for_short_segment[1:]) / 2

    # Load a hdf file.
    xh1 = load_timeseries(file_foreground, group=f'H1/{timestamps.start_time_str}')
    xl1 = load_timeseries(file_foreground, group=f'L1/{timestamps.start_time_str}')

    # Estimate PSD
    psd_h = pycbc.psd.welch(xh1, seg_len=sp.fftlength, seg_stride=sp.overlaplength, avg_method='median-mean')
    psd_h_interp = pycbc.psd.interpolate(psd_h, delta_f=1.0 / sp.duration)
    psd_l = pycbc.psd.welch(xl1, seg_len=sp.fftlength, seg_stride=sp.overlaplength, avg_method='median-mean')
    psd_l_interp = pycbc.psd.interpolate(psd_l, delta_f=1.0 / sp.duration)
    print(f"[PID {os.getpid()}] Processing {timestamps.start_time}: PSD estimated")

    dataidx = 0
    for tc in tclist_for_short_segment:
        tini = tc - sp.duration / 2
        tfin = tc + sp.duration / 2
        if (timestamps.start_time < tini) and (tfin < timestamps.end_time):
            # Slice the data and window
            strain_h = xh1.time_slice(tini, tfin) * window
            strain_l = xl1.time_slice(tini, tfin) * window

            # Calculate SNR
            for i in range(sp.height_input):
                rho_h = matched_filter(template_bank[i], strain_h, psd=psd_h_interp, low_frequency_cutoff=sp.low_frequency_cutoff)
                rho_l = matched_filter(template_bank[i], strain_l, psd=psd_l_interp, low_frequency_cutoff=sp.low_frequency_cutoff)
                snrlist[0, i] = torch.from_numpy(abs(rho_h).numpy())[sp.kcrop_left: sp.kcrop_right]
                snrlist[1, i] = torch.from_numpy(abs(rho_l).numpy())[sp.kcrop_left: sp.kcrop_right]

            dataavg = make_snrmap_coarse(snrlist, sp.kfilter).to(torch.float32)
            torchfilename = f'{outdir}/inputs_{timestamps.start_time_str}_{int(sp.duration):d}_{dataidx:d}.pth'
            torch.save(dataavg, torchfilename)
            # print(f'[PID {os.getpid()}] Torch file ({torchfilename}) is saved.')
            dataidx += 1


def main(args):

    outdir = args.outdir
    print(f'Results will be output the directory {outdir}')
    if_not_exist_makedir(outdir)
    file_foreground = args.foreground
    print(f'Foreground file = {file_foreground}')
    file_injection = args.injection
    print(f'Injection file = {file_injection}')
    nstart = args.nstart
    nend = args.nend

    sp = SignalProcessingParameters(
        duration=16,
        tsegment=1,
        fs=2048,
        low_frequency_cutoff=20.0,
        tukey_alpha=1.0 / 16.0,
        tfft=4.0,
        width_input=256,
        height_input=256
    )

    # Get timestamp informations
    segment_timestamp_list = get_timestamps(file_foreground, file_injection, sp, nstart, nend)

    # Make a template bank
    approximant_tmp = 'IMRPhenomXPHM'
    mcmin_tmp = 5.0
    mcmax_tmp = 50.0
    ngrid_mc = sp.height_input
    mclist = np.logspace(np.log10(mcmin_tmp), np.log10(mcmax_tmp), ngrid_mc, endpoint=True)
    eta = 0.25
    a1 = 0.0
    a2 = 0.0
    template_bank = []
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
        template_bank.append(hp_fd)

    # Run the main code
    with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
        futures = [executor.submit(matchedfilter_core, file_foreground, template_bank, outdir, sp, ts, args.offevent) for ts in segment_timestamp_list]
        results = [f.result() for f in futures]
    print(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate matched filter images by using MDC dataset')
    parser.add_argument('--outdir', type=str, help='Directory where the data will be saved.')
    parser.add_argument('--foreground', type=str, help='Foreground file. This must be HDF5. See MDC wiki for the detail.')
    parser.add_argument('--injection', type=str, default=None, help='Injection file. This must be HDF5. See MDC wiki for the detail.')
    parser.add_argument('--offevent', action='store_true', help='Use the data where the signal is not injected.')
    parser.add_argument('--nstart', type=int, default=0, help='(To be discarded) Start index of the segment')
    parser.add_argument('--nend', type=int, default=129, help='(To be discarded) End index of the segment')
    args = parser.parse_args()
    main(args)
