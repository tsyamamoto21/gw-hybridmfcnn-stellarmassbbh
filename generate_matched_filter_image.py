#!/usr/bin/env python
import re
import os
import math
import h5py
import pickle
import subprocess
import numpy as np
from scipy.signal.windows import tukey
import torch
from pycbc.inject import InjectionSet
from pycbc.waveform import get_fd_waveform
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta
from pycbc.filter import highpass, matched_filter, sigmasq
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.detector import Detector
from pycbc.types import TimeSeries
import pycbc.noise
from dl4longcbc.utils import if_not_exist_makedir
import concurrent.futures


INJECTION_TIME_STEP = 24
NINJECTION_PER_FILE = 1000


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
        self.dt = 1.0 / self.fs
        self.df = 1.0 / self.duration
        self.tlen = int(self.duration * self.fs)

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


def create_injections(nsample: int, config: str, gps_start_time: int, injection_file: str, force: bool = False):
    gps_end_time = nsample * INJECTION_TIME_STEP + gps_start_time
    cmd = ['pycbc_create_injections']
    cmd += ['--config-files', config]
    cmd += ['--gps-start-time', str(gps_start_time)]
    cmd += ['--gps-end-time', str(gps_end_time)]
    cmd += ['--time-step', str(INJECTION_TIME_STEP)]
    cmd += ['--time-window', str(0)]
    # cmd += ['--seed', str(seed)]
    cmd += ['--output-file', injection_file]
    if force:
        cmd += ['--force']
    subprocess.call(cmd)


def calculate_snr(injection_file: str, psd, sp: SignalProcessingParameters, detectors: dict):
    pattern = 'injections_(\\d+).hdf'
    basename = os.path.basename(injection_file)
    dirname = os.path.dirname(injection_file)
    result = re.match(pattern, basename)
    idx_file = result.group(1)

    storedata = []
    rhodict = {k: [] for k in detectors.keys()}
    rhodict['total'] = []
    with h5py.File(injection_file, 'r') as fo:
        ninj = len(fo['tc'])
        for idx in range(ninj):
            storedata.append([])
            storedata[idx].append(idx)
            # Parameters
            m1 = fo['mass1'][idx]
            m2 = fo['mass2'][idx]
            distance = fo['distance'][idx]
            inclination = fo['inclination'][idx]
            ra = fo['ra'][idx]
            dec = fo['dec'][idx]
            pol = fo['polarization'][idx]
            tc = fo['tc'][idx]
            # Calculate SNR
            hp, hc = get_fd_waveform(
                approximant='IMRPhenomXPHM',
                mass1=m1,
                mass2=m2,
                distance=distance,
                inclination=inclination,
                delta_f=sp.df,
                f_lower=sp.low_frequency_cutoff)
            rhosq_tot = 0
            for ifo in ['H1', 'L1']:
                fp, fc = detectors[ifo].antenna_pattern(ra, dec, pol, tc)
                h = fp * hp + fc * hc
                rhosq_ifo = sigmasq(h, psd, low_frequency_cutoff=sp.low_frequency_cutoff, high_frequency_cutoff=sp.high_frequency_cutoff)
                rhosq_tot += rhosq_ifo
                rhodict[ifo].append(math.sqrt(rhosq_ifo))
            rhotot = math.sqrt(rhosq_tot)
            rhodict['total'].append(rhotot)
            # Store SNRs
            storedata[idx].append(f'input_{idx_file}_{idx:d}.pth')
            for ifo in ['H1', 'L1']:
                storedata[idx].append(rhodict[ifo][idx])
            storedata[idx].append(rhodict['total'][idx])
    with open(f'{dirname}/snrlist_{idx_file}.pkl', 'wb') as f:
        pickle.dump(storedata, f)


def generate_matchedfilter_image(outdir: str, fileidx: int, template_bank: dict, sp: SignalProcessingParameters, psd, detectors: dict, noiseonly=False, nonoise=False):

    injection_file = f'{outdir}/injections_{fileidx:d}.hdf'
    injector = InjectionSet(injection_file)
    injtable = injector.table

    # SNR image array
    snrlist = torch.zeros((2, sp.height_input, sp.width_before_smearing), requires_grad=False, dtype=torch.complex128)
    # Tukey window
    window = tukey(sp.mfdatalength, sp.tukey_alpha)
    for idx in range(len(injtable)):
        # Generate strain
        tc = injtable[idx]['tc']
        for idx_detector, (k, ifo) in enumerate(detectors.items()):
            if nonoise:
                strain = TimeSeries(np.zeros((sp.tlen,)), delta_t=sp.dt)
            else:
                strain = pycbc.noise.noise_from_psd(sp.tlen, sp.dt, psd)
            strain.start_time = tc - (sp.duration / 2)
            if noiseonly:
                pass
            else:
                injector.apply(strain, k)
            strain = highpass(strain, 15.0)
            # Estimate PSD
            if nonoise:
                psd_interp = pycbc.psd.interpolate(psd, delta_f=1.0 / sp.duration)
            else:
                psd_estimated = pycbc.psd.welch(strain, seg_len=sp.fftlength, seg_stride=sp.overlaplength, avg_method='median-mean')
                psd_interp = pycbc.psd.interpolate(psd_estimated, delta_f=1.0 / sp.duration)

            # Calculate SNR with template bank
            for i in range(sp.height_input):
                rho = matched_filter(template_bank['template'][i], strain * window, psd=psd_interp, low_frequency_cutoff=sp.low_frequency_cutoff)
                # snrlist[idx_detector, i] = torch.from_numpy(abs(rho).numpy())[sp.kcrop_left: sp.kcrop_right]
                snrlist[idx_detector, i] = torch.from_numpy(rho.numpy())[sp.kcrop_left: sp.kcrop_right]

            # If necessary, strain is saved
            strainfilename = f'{outdir}/strain_{fileidx:d}_{idx:d}_{k}.pkl'
            with open(strainfilename, 'wb') as fo:
                pickle.dump(strain, fo)

        # Smearing and storing the data
        torchfilename = f'{outdir}/input_{fileidx:d}_{idx:d}.pth'
        # dataavg = make_snrmap_coarse(snrlist, sp.kfilter).to(torch.float32)
        # torch.save(dataavg, torchfilename)
        torch.save(snrlist, torchfilename)


def main(args):
    outdir = args.outdir
    if_not_exist_makedir(outdir)
    ndata = args.ndata
    ninjfile = int(math.ceil(ndata / NINJECTION_PER_FILE))
    print(f'{ninjfile=}')
    if args.noiseonly:
        label = 'noise'
    else:
        label = 'cbc'
    if_not_exist_makedir(f'{outdir}/{label}')

    # Strain parameters
    ifonamelist = ['H1', 'L1']
    ifodict = {ifoname: Detector(ifoname) for ifoname in ifonamelist}

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

    # PSD parameters
    delta_f_psd = 1.0 / sp.duration
    psdparam = {
        'length': int(sp.fs / 2 / delta_f_psd) + 1,
        'delta_f': delta_f_psd,
        'low_freq_cutoff': 5.0
    }
    psd_analytic = aLIGOZeroDetHighPower(**psdparam)

    # Make a template bank
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

    # Create injection parameters
    gps_start_time = args.starttime
    for n in range(ninjfile):
        injection_file = f'{args.outdir}/{label}/injections_{n + args.offset:d}.hdf'
        if n == ninjfile - 1:
            if ndata % NINJECTION_PER_FILE == 0:
                nsample = NINJECTION_PER_FILE
            else:
                nsample = int(ndata % NINJECTION_PER_FILE)
        else:
            nsample = NINJECTION_PER_FILE
        create_injections(nsample, args.config, gps_start_time, injection_file, force=args.force)
        gps_start_time += nsample * INJECTION_TIME_STEP

    # Calculate SNR
    snrcalculate_list = []
    for n in range(ninjfile):
        if not os.path.exists(f'{args.outdir}/{label}/snrlist_{n + args.offset:d}.hdf'):
            snrcalculate_list.append(n + args.offset)
    if len(snrcalculate_list) != 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
            futures = [executor.submit(calculate_snr, f'{args.outdir}/{label}/injections_{n:d}.hdf', psd_analytic, sp, ifodict) for n in snrcalculate_list]
            results = [f.result() for f in futures]
        print('SNR calculation: ', results)

    # Generate matched filter images
    if args.parameteronly:
        pass
    else:
        # def generate_matchedfilter_image(outdir: str, fileidx: int, template_bank: list, sp: SignalProcessingParameters, psd, detectors: dict, noiseonly=False):
        # Run the main code
        with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
            futures = [executor.submit(generate_matchedfilter_image, f'{outdir}/{label}/', n, template_bank, sp, psd_analytic, ifodict, args.noiseonly, args.nonoise) for n in range(args.offset, args.offset + ninjfile)]
            results = [f.result() for f in futures]
        print('Generate matched filter images: ', results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate matched filter images.')
    parser.add_argument('--outdir', type=str, help='Directory name including `train` or `validate` or `test`.')
    parser.add_argument('--ndata', type=int, help='Data number')
    parser.add_argument('--config', type=str, help='Configure file of injection')
    # parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--starttime', type=int, help='Injection start GPS time.')
    parser.add_argument('--noiseonly', action='store_true', help='If true, no GW signal are injected.')
    parser.add_argument('--nonoise', action='store_true', help='If true, no noise are injected.')
    parser.add_argument('--parameteronly', action='store_true', help='If true, no foreground is generated.')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()

    # Check arguments consistency
    if args.noiseonly * args.nonoise:
        raise ValueError("noiseonly and nonoise cannot be true at the same time.")

    main(args)
