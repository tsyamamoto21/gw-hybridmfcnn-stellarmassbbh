#!/usr/bin/env python
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


def make_snrmap_coarse(snrmap, kfilter):
    nc, nx, ny = snrmap.shape
    ny_coarse = ny // kfilter
    snrmap_coarse = torch.zeros((nc, nx, ny_coarse), dtype=torch.float32)
    for i in range(ny_coarse):
        snrmap_coarse[:, :, i] = torch.sqrt(torch.mean(snrmap[:, :, i * kfilter: (i + 1) * kfilter]**2, dim=-1))
    return snrmap_coarse


def main(args):

    outdir = args.outdir
    file_foreground = args.foreground
    file_injection = args.injection

    # Parameters for template matching
    duration = 16
    tsegment = 1
    fs = 2048
    fhigh = fs / 2
    low_frequency_cutoff = 20.0
    fftlength = int(4 * fs)
    overlap_length = int(fftlength / 2)
    window = tukey(int(duration * fs), alpha=1.0 / 16.0)

    approximant_tmp = 'IMRPhenomXPHM'
    mcmin_tmp = 5.0
    mcmax_tmp = 50.0
    ngrid_mc = 256
    mclist = np.logspace(np.log10(mcmin_tmp), np.log10(mcmax_tmp), ngrid_mc, endpoint=True)
    eta = 0.25
    a1 = 0.0
    a2 = 0.0

    # Image parameters
    # height = ngrid_mc
    width_input = 256
    kfilter = int(fs * tsegment / width_input)
    kcrop_left = int(fs * (duration / 2 - 3 * tsegment / 4))
    kcrop_right = int(fs * (duration / 2 + 3 * tsegment / 4))
    width_before_smearing = kcrop_right - kcrop_left

    # Make a template bank
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
            'f_lower': low_frequency_cutoff,
            'delta_f': 1.0 / duration,
            'f_final': fhigh
        }

        hp_fd, _ = get_fd_waveform(**params_tmp)
        template_bank.append(hp_fd)

    # Get start time and end time
    with h5py.File(file_foreground, 'r') as fo:
        x = fo['H1'].keys()
        start_time_strlist = [xsample for xsample in x]
        start_time_list = [int(s) for s in start_time_strlist]
        nsegment = len(start_time_list)
        end_time_list = [int(len(fo['H1'][start_time_strlist[i]]) / fs) + start_time_list[i] for i in range(nsegment)]

    # Get the injection time stamps
    with h5py.File(file_injection, 'r') as fo:
        tclist_from_hdf5 = fo['tc'][:]
    # Assign a segment to an injection
    tclist = [[] for _ in range(nsegment)]
    for idx in range(len(tclist_from_hdf5)):
        for n in range(nsegment):
            if (start_time_list[n] <= tclist_from_hdf5[idx]) * (tclist_from_hdf5[idx] <= end_time_list[n]):
                tclist[n].append(tclist_from_hdf5[idx])

    # Run the main code
    snrlist = torch.zeros((2, ngrid_mc, width_before_smearing), requires_grad=False)
    dataidx = 0
    for n in range(nsegment):
        # Set time stampes
        start_time_str = start_time_strlist[n]
        start_time = start_time_list[n]
        end_time = end_time_list[n]
        tclist_for_short_segment = tclist[n]
        if args.offevent:
            tclist_for_short_segment = np.array(tclist_for_short_segment)
            tclist_for_short_segment = (tclist_for_short_segment[:-1] + tclist_for_short_segment[1:]) / 2

        # Load a hdf file.
        xh1 = load_timeseries(file_foreground, group=f'H1/{start_time_str}')
        xl1 = load_timeseries(file_foreground, group=f'L1/{start_time_str}')

        # Estimate PSD
        psd_h = pycbc.psd.welch(xh1, seg_len=fftlength, seg_stride=overlap_length, avg_method='median-mean')
        psd_h_interp = pycbc.psd.interpolate(psd_h, delta_f=1.0 / duration)
        psd_l = pycbc.psd.welch(xl1, seg_len=fftlength, seg_stride=overlap_length, avg_method='median-mean')
        psd_l_interp = pycbc.psd.interpolate(psd_l, delta_f=1.0 / duration)

        for tc in tclist_for_short_segment:
            tini = tc - duration / 2
            tfin = tc + duration / 2
            if (start_time < tini) and (tfin < end_time):
                print(tc)
                # Slice the data and window
                strain_h = xh1.time_slice(tini, tfin) * window
                strain_l = xl1.time_slice(tini, tfin) * window

                # Calculate SNR
                # rholist = []
                for i in range(ngrid_mc):
                    rho_h = matched_filter(template_bank[i], strain_h, psd=psd_h_interp, low_frequency_cutoff=low_frequency_cutoff)
                    rho_l = matched_filter(template_bank[i], strain_l, psd=psd_l_interp, low_frequency_cutoff=low_frequency_cutoff)
                    # snrlist[i] = torch.from_numpy(((abs(rho_h)**2 + abs(rho_l)**2) ** 0.5).numpy().astype(np.float32))
                    snrlist[0, i] = torch.from_numpy(abs(rho_h).numpy())[kcrop_left: kcrop_right]
                    snrlist[1, i] = torch.from_numpy(abs(rho_l).numpy())[kcrop_left: kcrop_right]

                dataavg = make_snrmap_coarse(snrlist, kfilter).to(torch.float32)
                torch.save(dataavg, f'{outdir}/inputs_{int(duration):d}_{dataidx:d}.pth')
                dataidx += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate matched filter images by using MDC dataset')
    parser.add_argument('--outdir', type=str, help='Directory where the data will be saved.')
    parser.add_argument('--foreground', type=str, help='Foreground file. This must be HDF5. See MDC wiki for the detail.')
    parser.add_argument('--injection', type=str, help='Injection file. This must be HDF5. See MDC wiki for the detail.')
    parser.add_argument('--offevent', action='store_true', help='Use the data where the signal is not injected.')
    args = parser.parse_args()
    main(args)
