#!/usr/bin/env python
import os
import numpy as np
from scipy.signal.windows import tukey
import torch
from pycbc.waveform import get_fd_waveform, get_td_waveform
import pycbc.conversions as pc
from pycbc.filter import highpass, matched_filter
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
import pycbc.noise
from dl4longcbc.utils import if_not_exist_makedir
import concurrent.futures


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

        # Noise image
        self.width_noise_image = 4 * 4096


def _condition_m1_larger_than_m2(m1sample, m2sample):
    mask = m1sample < m2sample
    m1_tmp = m1sample.copy()
    m1sample[mask] = m2sample[mask]
    m2sample[mask] = m1_tmp[mask]
    return m1sample, m2sample


def generate_signal_matchedfilter_image(outdir: str, fileidx: int, template_bank: dict, sp: SignalProcessingParameters, psd, detectors: dict):

    # Generate injection signal
    mmin, mmax = 10.0, 50.0
    m1 = np.random.uniform(mmin, mmax)
    m2 = np.random.uniform(mmin, mmax)
    if m2 > m1:
        m2tmp = m2
        m2 = m1
        m1 = m2tmp
    hp_inj, hc_inj = get_td_waveform(
        approximant='IMRPhenomXPHM',
        mass1=m1,
        mass2=m2,
        f_lower=sp.low_frequency_cutoff,
        delta_t=sp.dt
    )

    # Inject signal
    strain_p = TimeSeries(np.zeros((sp.tlen,)), delta_t=sp.dt, epoch=-sp.duration // 2)
    strain_c = TimeSeries(np.zeros((sp.tlen,)), delta_t=sp.dt, epoch=-sp.duration // 2)
    strain_p = strain_p.inject(hp_inj)
    strain_c = strain_c.inject(hc_inj)

    # Prepare SNR image array
    snrlist = torch.zeros((2, sp.height_input, sp.width_before_smearing), requires_grad=False, dtype=torch.complex128)
    # PSD
    psd_interp = pycbc.psd.interpolate(psd, delta_f=1.0 / sp.duration)
    # Calculate SNR with template bank
    for i in range(sp.height_input):
        rho_p = matched_filter(template_bank['template'][i], strain_p, psd=psd_interp, low_frequency_cutoff=sp.low_frequency_cutoff)
        rho_c = matched_filter(template_bank['template'][i], strain_c, psd=psd_interp, low_frequency_cutoff=sp.low_frequency_cutoff)
        snrlist[0, i] = torch.from_numpy(rho_p.numpy())[sp.kcrop_left: sp.kcrop_right]
        snrlist[1, i] = torch.from_numpy(rho_c.numpy())[sp.kcrop_left: sp.kcrop_right]

    # # If necessary, strain is saved
    # strainfilename = f'{outdir}/strain_{fileidx:d}_{idx:d}_{k}.pkl'
    # with open(strainfilename, 'wb') as fo:
    #     pickle.dump(strain, fo)

    # Save the data
    torchfilename = os.path.join(outdir, f'signalmf_{fileidx:d}.pth')
    torch.save(snrlist, torchfilename)


def generate_noise_matchedfilter_image(outdir: str, fileidx: int, template_bank: dict, sp: SignalProcessingParameters, psd, detectors: dict):

    # SNR image array
    snrlist = torch.zeros((sp.height_input, sp.width_noise_image), requires_grad=False, dtype=torch.complex128)
    # Tukey window
    window = tukey(sp.mfdatalength, sp.tukey_alpha)
    # Generate strain
    strain = pycbc.noise.noise_from_psd(sp.tlen, sp.dt, psd)
    strain = highpass(strain, 15.0)
    psd_estimated = pycbc.psd.welch(strain, seg_len=sp.fftlength, seg_stride=sp.overlaplength, avg_method='median-mean')
    psd_interp = pycbc.psd.interpolate(psd_estimated, delta_f=1.0 / sp.duration)

    # Calculate SNR with template bank
    for i in range(sp.height_input):
        rho = matched_filter(template_bank['template'][i], strain * window, psd=psd_interp, low_frequency_cutoff=sp.low_frequency_cutoff)
        snrlist[i] = torch.from_numpy(rho.numpy())

    # # If necessary, strain is saved
    # strainfilename = f'{outdir}/strain_{fileidx:d}_{idx:d}_{k}.pkl'
    # with open(strainfilename, 'wb') as fo:
    #     pickle.dump(strain, fo)

    # Save the data
    torchfilename = os.path.join(outdir, f'noisemf_{fileidx:d}.pth')
    torch.save(snrlist, torchfilename)


def main(args):
    outdir = args.outdir
    if_not_exist_makedir(outdir)
    ndata = args.ndata
    if args.noise:
        label = 'noise'
    elif args.signal:
        label = 'cbc'
    else:
        raise ValueError('Somethig wrong at args.noise and/or args.signal')
    if_not_exist_makedir(os.path.join(outdir, label))

    # Strain parameters
    sp = SignalProcessingParameters(
        duration=16,
        tsegment=1,
        fs=2048,
        low_frequency_cutoff=20.0,
        tukey_alpha=1.0 / 16.0,
        tfft=4.0,
        width_input=256,
        height_input=256,
        # noise_mf_duration=4,
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
        mass1 = pc.mass1_from_mchirp_eta(mclist[i], eta)
        mass2 = pc.mass2_from_mchirp_eta(mclist[i], eta)
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

    # Generate matched filter images
    with concurrent.futures.ProcessPoolExecutor(max_workers=48) as executor:
        if args.noise:
            futures = [executor.submit(generate_noise_matchedfilter_image, os.path.join(outdir, label), n, template_bank, sp, psd_analytic, args.noiseonly, args.nonoise) for n in range(args.offset, args.offset + ndata)]
        elif args.signal:
            futures = [executor.submit(generate_signal_matchedfilter_image, os.path.join(outdir, label), n, template_bank, sp, psd_analytic, args.noiseonly, args.nonoise) for n in range(args.offset, args.offset + ndata)]
        else:
            ValueError('Something wrong with --noise and/or --signal.')
        results = [f.result() for f in futures]
    print('Generate matched filter images: ', results)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate matched filter images.')
    parser.add_argument('--outdir', type=str, help='Directory name including `train` or `validate` or `test`.')
    parser.add_argument('--ndata', type=int, help='Data number')
    parser.add_argument('--starttime', type=int, help='Injection start GPS time.')
    parser.add_argument('--noise', action='store_true', help='Noise mf image')
    parser.add_argument('--signal', action='store_true', help='Signal mf image')
    parser.add_argument('--offset', type=int, default=0)
    args = parser.parse_args()

    # Check arguments consistency
    if args.noise * args.signal:
        raise ValueError("noise and signal cannot be true at the same time.")
    elif not (args.noise or args.signal):
        raise ValueError("Specify --noise or --signal")

    main(args)
