#!/usr/bin/env python
import os
import time
import h5py
import numpy as np
from scipy.signal.windows import tukey, hann
import logging
import argparse
from pathlib import Path
import torch
from torch.nn.functional import interpolate as torch_interpolate
from omegaconf import OmegaConf
from pycbc.types import load_timeseries
from pycbc.waveform import get_fd_waveform
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta
from dl4longcbc.net import instantiate_neuralnetwork


class MDCResultTriplet:
    def __init__(self):
        self.time = []
        self.stat = []
        self.var = []
    
    def __len__(self):
        return len(self.time)

    def add(self, t, s, v):
        self.time.append(float(t))
        self.stat.append(float(s))
        self.var.append(v)

    def dump(self, outfile):
        with h5py.File(outfile, "w") as f:
            f.create_dataset("time", data=np.array(self.time), compression="gzip")
            f.create_dataset("stat", data=np.array(self.stat), compression="gzip")
            f.create_dataset("var", data=np.array(self.var), compression="gzip")

    def cluster_triggers(self):
        if len(self) < 2:
            print("There is no triggers to be clustered.")
            outresults = self
        else:
            outresults = MDCResultTriplet()
            trigger_list = []
            stat_list = []
            ti_buf = self.time[0]
            si_buf = [self.stat[0]]
            vi_buf = self.var[0]
            trigger_buf = [ti_buf - vi_buf, ti_buf + vi_buf]
            for idx in range(1, len(self)):
                ti = self.time[idx]
                si = self.stat[idx]
                vi = self.var[idx]
                if self._is_trigger_included_segment(ti, vi, trigger_buf):
                    trigger_buf[1] = ti + vi
                    si_buf.append(si)
                else:
                    trigger_list.append(trigger_buf)
                    stat_list.append(np.max(si_buf))
                    trigger_buf = [ti - vi, ti + vi]
                    si_buf = [si]
            trigger_list.append(trigger_buf)
            stat_list.append(np.max(si_buf))
            # Newly add the triggers
            for li, si in zip(trigger_list, stat_list):
                outresults.add((li[1] + li[0]) / 2.0, si, (li[1] - li[0]) / 2.0)
            return outresults

    def _is_trigger_included_segment(self, ti, vi, segment):
        flg = False
        if ti - vi <= segment[1]:
            flg = True
        return flg


class SignalProcessingParameters:
    def __init__(self, duration, fs, low_frequency_cutoff, tfft, tukey_alpha, tnnw, height_input, kappa):
        # Signal properties
        self.duration = duration
        self.fs = fs
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = self.fs / 2
        self.dt = 1.0 / self.fs
        self.df = 1.0 / self.duration
        self.tlen = int(self.duration * self.fs)
        self.flen = self.tlen // 2 + 1

        # FFT for PSD estimation
        self.tseg = tfft
        self.seg_len = int(self.tseg * self.fs)
        self.toverlap = self.tseg / 2
        self.seg_overlap = int(self.toverlap * self.fs)
        self.tstride = self.tseg - self.toverlap
        self.seg_stride = self.seg_len - self.seg_overlap
        self.fftwindow = torch.from_numpy(hann(self.seg_len)).to(dtype=torch.float32, device='cuda')
        self.W = torch.mean(self.fftwindow ** 2)

        # MF params
        self.kappa = kappa
        self.tukey_alpha = tukey_alpha
        self.window = tukey(self.tlen, self.tukey_alpha)
        self.window_torch = torch.from_numpy(self.window).to(dtype=torch.float32, device='cuda')

        # Image properties
        self.tnnw = tnnw
        self.nnw_len = int(self.fs * self.tnnw)
        self.nnw_overlap = self.nnw_len // 2
        self.width_input = self.nnw_len
        self.height_input = height_input

        # Get frequency mask
        self.fmask = self._get_frequency_mask()
        self.fmask_torch = torch.from_numpy(self.fmask).to(dtype=torch.float32, device='cuda')

    def _get_frequency_mask(self) -> np.ndarray:
        # Get mask
        fmask = np.zeros((self.flen))
        kmin = int(self.low_frequency_cutoff / self.df)
        kmax = int(self.high_frequency_cutoff / self.df)
        if kmax > self.flen:
            print(f"kmax ({kmax}) is larger than flen ({self.flen}).")
        fmask[kmin: kmax] = 1.0
        return fmask


def fold_tensor(strain: torch.Tensor, sp: SignalProcessingParameters) -> torch.Tensor:
    '''
    strain should have a size of (2, 1, N)
    Output has a size of (nseg, 2, 1, sp.seg_len)
    '''
    xlen = strain.shape[2]
    _, nseg_mod = divmod((xlen - sp.tlen // 2), sp.tlen // 2)
    flg_add_last_part = nseg_mod != 0

    strain_folded = strain.unfold(dimension=2, size=sp.tlen, step=sp.tlen // 2).permute(2, 0, 1, 3)
    if flg_add_last_part:
        strain_folded = torch.cat([strain_folded, strain[:, :, -sp.tlen:].unsqueeze(0)], dim=0)
    return strain_folded


def median_bias(ns):
    ans = 1
    for i in range(1, (ns - 1) // 2 + 1):
        ans += 1.0 / (2 * i + 1) - 1.0 / (2 * i)
    return ans


def torch_median(inputs: torch.Tensor) -> torch.Tensor:
    '''
    inputs must have the shape of (2, C, N)
    median will be taken along with the second dimension (C,)
    '''
    C = inputs.shape[1]
    if C % 2 == 1:
        # 要素数が奇数の場合、通常のtorch.medianを使用
        return torch.median(inputs, dim=1, keepdim=True).values
    else:
        # 要素数が偶数の場合、中央の2つの要素の平均を計算
        sorted_x = torch.sort(inputs, dim=1).values
        mid_low = sorted_x[:, C // 2 - 1].view(2, 1, -1)
        mid_high = sorted_x[:, C // 2].view(2, 1, -1)
        return (mid_low + mid_high) / 2


def torch_estimate_psd(strain: torch.Tensor, sp: SignalProcessingParameters) -> torch.Tensor:
    assert strain.shape[0] == 2, "strain should include 2 channels, (L and H)"

    _, nseg_mod = divmod((sp.tlen - sp.seg_overlap), sp.seg_stride)
    kstart = nseg_mod // 2
    kend = strain.shape[1] - nseg_mod // 2
    strain_folded = strain[:, kstart:kend].unfold(dimension=1, size=sp.seg_len, step=sp.seg_stride)
    nsft = strain_folded.shape[1]
    nsft_odd = nsft // 2 + 1
    nsft_even = nsft // 2
    alpha_odd = median_bias(nsft_odd)
    alpha_even = median_bias(nsft_even)

    # FFT
    strain_folded_fd = torch.fft.rfft(strain_folded * sp.fftwindow, dim=-1) * sp.dt

    # Calculate PSD
    P_all = 2.0 * torch.abs(strain_folded_fd) ** 2.0 / sp.tseg / sp.W
    # Calculate PSD by median-mean
    P_odd = torch_median(P_all[:, ::2])
    P_even = torch_median(P_all[:, 1::2])
    psd = (P_even / alpha_even + P_odd / alpha_odd) / 2.0

    # interpolate
    psd_interp = torch_interpolate(psd, size=(sp.flen,), mode='linear', align_corners=True)
    return psd_interp


# normalize template
def torch_sigmasq(h2_mat: torch.Tensor, psd: torch.Tensor, sp: SignalProcessingParameters) -> torch.Tensor:
    assert h2_mat.ndim == 3, "h2_mat does not have dim of 3."
    assert h2_mat.shape[0] == 1, f"shape error: {h2_mat.shape=}"
    assert psd.ndim == 3, f"shape error: {psd.shape=}"
    assert psd.shape[0] == 2, f"shape error: {psd.shape=}"
    assert psd.shape[1] == 1, f"shape error: {psd.shape=}"
    assert h2_mat.shape[2] == psd.shape[2], "template in frequency domain and PSD does not have the same length."

    ntemp = h2_mat.shape[1]
    return 4.0 * torch.trapezoid(h2_mat / psd, dx=sp.df, dim=-1).reshape(2, ntemp, 1)


def torch_matched_filter(strain_td: torch.Tensor, hconj_mat: torch.Tensor, psd: torch.Tensor, norm: torch.Tensor, sp: SignalProcessingParameters) -> torch.Tensor:
    '''
    * hconj must be multiplied by fmask
    * norm = sigmasq = (h|h)
    '''
    assert hconj_mat.ndim == 3, "hp2_mat does not have dim of 3."
    assert hconj_mat.shape[0] == 1, f"shape error: {hconj_mat.shape=}"
    assert strain_td.ndim == 3, "strain_td does not have dim of 3"
    assert strain_td.shape[0] == 2, f"shape error: {strain_td.shape=}"
    assert strain_td.shape[1] == 1, f"shape error: {strain_td.shape=}"
    assert psd.ndim == 3, f"shape error: {psd.shape=}"
    assert psd.shape[0] == 2, f"shape error: {psd.shape=}"
    assert psd.shape[1] == 1, f"shape error: {psd.shape=}"
    assert hconj_mat.shape[2] == psd.shape[2], "template in frequency domain and PSD does not have the same length."

    strain_fd = torch.fft.rfft(strain_td * sp.window_torch, dim=-1) * sp.dt
    return 4.0 * torch.fft.ifft(strain_fd * hconj_mat / psd, sp.tlen, dim=-1) / sp.dt / torch.sqrt(norm)


def main(args):

    # Time series property
    logging.info('Set the time series property.')
    sp = SignalProcessingParameters(
        duration=16.0,
        fs=2048,
        low_frequency_cutoff=20.0,
        tukey_alpha=1.0 / 16.0,
        tfft=4.0,
        tnnw=1.0,
        height_input=256,
        kappa=1e+22
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
    # Get templates
    template = torch.zeros((1, ngrid_mc, sp.flen), dtype=torch.complex64, device='cuda')
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
        # template_cache.append(hp_fd)
        template[0, i] = torch.from_numpy(hp_fd.numpy()).to(dtype=torch.complex64, device='cuda') * sp.fmask_torch * sp.kappa
    template_conj = template.conj()
    template_2 = (template * template_conj).real

    # Load a trained neural network
    logging.info('Loading the trained neural network.')
    logging.warning('!!! To be implemented !!!')

    # Get segments
    logging.info('Get segments')
    with h5py.File(args.inputfile, 'r') as file:
        list_start_time = [int(k) for k in file['H1'].keys()]
    list_start_time.sort()

    # Load trained netrowk
    config_file = os.path.join(args.modeldir, 'config_train.yaml')
    model_file = os.path.join(args.modeldir, 'model.pth')
    config_nn = OmegaConf.load(config_file)
    model = instantiate_neuralnetwork(config_nn)
    model.load_state_dict(torch.load(model_file, weights_only=True))
    model = model.to('cuda')
    model.eval()

    # Prepare result container
    mdc_results = MDCResultTriplet()

    for start_time in list_start_time:
        # Load strains
        logging.info(f'Start time = {start_time}: Loading strains')
        h1_ts = load_timeseries(args.inputfile, group=f'H1/{start_time}')
        l1_ts = load_timeseries(args.inputfile, group=f'L1/{start_time}')
        duration = h1_ts.duration
        start_time_gps = h1_ts.start_time

        # Into torch tensor
        strain = torch.zeros((2, 1, len(h1_ts)), dtype=torch.float32, device='cuda')
        strain[0, 0] = torch.from_numpy(h1_ts.numpy()).to(dtype=torch.float32, device='cuda') * sp.kappa
        strain[1, 0] = torch.from_numpy(l1_ts.numpy()).to(dtype=torch.float32, device='cuda') * sp.kappa

        # Fold strain into 16s segments
        strain_folded = fold_tensor(strain, sp)
        Npsdsegs = strain_folded.shape[0]

        # Prepare empty tensors
        logging.info(f'Start time = {start_time}: Making SNR maps')

        tik = time.time()
        for idxpsd in range(Npsdsegs):
            # Estimate PSD
            psd = torch_estimate_psd(strain_folded[idxpsd, :, 0], sp)
            # Normalization factor
            sigmasq_torch = torch_sigmasq(template_2, psd, sp)
            # Matched filter
            matched_filter_torch = torch_matched_filter(strain_folded[idxpsd], template_conj, psd, sigmasq_torch, sp)

            # Process MF outputs by neural network
            if idxpsd == Npsdsegs - 1:
                kstart = sp.tlen // 4
                kend = sp.tlen * 3 // 4 + sp.nnw_overlap
                mfwindow_tstart = start_time_gps + idxpsd * sp.tstride - (duration % sp.tstride)
            else:
                kstart = sp.tlen // 4
                kend = sp.tlen * 3 // 4 + sp.nnw_overlap
                mfwindow_tstart = start_time_gps + idxpsd * sp.tstride
            matched_filter_torch_unfolded = matched_filter_torch[:, :, kstart: kend].unfold(dimension=2, size=sp.nnw_len, step=sp.nnw_overlap).permute(2, 0, 1, 3)

            # Process by neural network
            logging.info(f'Start time = {start_time}: Processing SNR maps by the neural network.')
            with torch.no_grad():
                output = model(matched_filter_torch_unfolded).to('cpu')

            # Get [time, stat, var]
            logging.info(f'Start time = {start_time}: Summarizing into [time, stat, var] triplets.')
            stat_all = output[:, 1] - output[:, 0]
            threshold = 1.0
            for i, stat in enumerate(stat_all):
                if stat >= threshold:
                    mdc_results.add(mfwindow_tstart + sp.tseg // 4 + (i + 1) * sp.tnnw / 2, stat, 0.5)

        tok = time.time()
        break
    logging.info(f'Elapsed time {tok - tik} seconds for {Npsdsegs} psdsegments')

    # Save result triples
    logging.info('Clustering the triggers.')
    mdc_results_clustered = mdc_results.cluster_triggers()
    logging.info('Saving result triples')
    mdc_results_clustered.dump(args.outputfile)
    logging.info('Result saved')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process MDC data by neural network.')
    parser.add_argument('-i', '--inputfile', type=str, required=True, help='hdf5 file of strain data.')
    parser.add_argument('-o', '--outputfile', type=str, required=True, help='hdf5 file to be output.')
    parser.add_argument('--modeldir', type=str, required=True, help='Directory where the trained neural network model is saved.')
    args = parser.parse_args()

    # Setup logging
    log_level = logging.INFO
    logging.basicConfig(format='%(levelname)-8s | %(asctime)s | %(message)s',
                        level=log_level, datefmt='%Y-%m-%d %H:%M:%S')

    assert os.path.exists(args.inputfile), f"Input file {args.inputfile} does not exist."
    assert Path(args.inputfile).suffix == '.hdf', f"Input file must be an hdf5 file. (Given {args.inputfile})"
    assert Path(args.outputfile).suffix == '.hdf', f"Output file must be an hdf5 file. (Given {args.outputfile})"

    main(args)
