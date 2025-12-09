#!/usr/bin/env python
import os
import time
import logging
import numpy as np
from scipy.signal.windows import tukey, hann
import torch
from torch.nn.functional import interpolate as torch_interpolate
from pycbc.waveform import get_fd_waveform, get_td_waveform
import pycbc.conversions as pc
from pycbc.filter import highpass
from pycbc.psd.analytical import aLIGOZeroDetHighPower
from pycbc.types import TimeSeries
import pycbc.noise
from gw_hybridmfcnn.utils import if_not_exist_makedir


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
        # self.fftwindow = torch.from_numpy(hann(self.seg_len)).to(dtype=torch.float32)
        self.W = torch.mean(self.fftwindow ** 2)

        # MF params
        self.kappa = kappa
        self.tukey_alpha = tukey_alpha
        self.window = tukey(self.tlen, self.tukey_alpha)
        self.window_torch = torch.from_numpy(self.window).to(dtype=torch.float32, device='cuda')
        # self.window_torch = torch.from_numpy(self.window).to(dtype=torch.float32)

        # Image properties
        self.tnnw = tnnw
        self.nnw_len = int(self.fs * self.tnnw)
        self.nnw_overlap = self.nnw_len // 2
        self.width_input = self.nnw_len
        self.height_input = height_input

        # Get frequency mask
        self.fmask = self._get_frequency_mask()
        self.fmask_torch = torch.from_numpy(self.fmask).to(dtype=torch.float32, device='cuda')
        # self.fmask_torch = torch.from_numpy(self.fmask).to(dtype=torch.float32)

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


def median_bias(ns: int) -> float:
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


def generate_signal_matchedfilter_image(outdir: str, fileidx: int, template_conj: torch.Tensor, template_2: torch.Tensor, sp: SignalProcessingParameters, psd: torch.Tensor):

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

    # Into torch tensor
    strain = torch.zeros((2, 1, len(strain_p)), dtype=torch.float32, device='cuda')
    strain[0, 0] = torch.from_numpy(strain_p.numpy() * sp.kappa).to(dtype=torch.float32, device='cuda')
    strain[1, 0] = torch.from_numpy(strain_c.numpy() * sp.kappa).to(dtype=torch.float32, device='cuda')
    # strain = torch.zeros((2, 1, len(strain_p)), dtype=torch.float32)
    # strain[0, 0] = torch.from_numpy(strain_p.numpy() * sp.kappa).to(dtype=torch.float32)
    # strain[1, 0] = torch.from_numpy(strain_c.numpy() * sp.kappa).to(dtype=torch.float32)

    # Normalization factor
    sigmasq_torch = torch_sigmasq(template_2, psd, sp)
    # Matched filter
    matched_filter_torch = torch_matched_filter(strain, template_conj, psd, sigmasq_torch, sp)

    # Save the data
    kstart = sp.tlen // 4
    kend = sp.tlen * 3 // 4
    torchfilename = os.path.join(outdir, f'signalmf_{fileidx:d}.pth')
    torch.save(matched_filter_torch[:, :, kstart: kend].to('cpu'), torchfilename)


def generate_noise_matchedfilter_image(outdir: str, fileidx: int, template_conj: torch.Tensor, template_2: torch.Tensor, sp: SignalProcessingParameters, psd):

    # Generate strain
    noisestrain = pycbc.noise.noise_from_psd(sp.tlen, sp.dt, psd)
    noisestrain = highpass(noisestrain, 15.0)
    strain = torch.ones((2, 1, len(noisestrain)), dtype=torch.float32, device='cuda')  # 2nd row is just dummy
    strain[0, 0] = torch.from_numpy(noisestrain.numpy() * sp.kappa).to(dtype=torch.float32, device='cuda')

    # Estimate PSD
    psd_torch = torch_estimate_psd(strain[:, 0], sp)
    # Normalization factor
    sigmasq_torch = torch_sigmasq(template_2, psd_torch, sp)
    # Matched filter
    matched_filter_torch = torch_matched_filter(strain, template_conj, psd_torch, sigmasq_torch, sp)

    # Save the data
    kstart = sp.tlen // 4
    kend = sp.tlen * 3 // 4
    torchfilename = os.path.join(outdir, f'noisemf_{fileidx:d}.pth')
    logging.debug(f'{matched_filter_torch.size()}')
    torch.save(matched_filter_torch[0, :, kstart: kend].to('cpu'), torchfilename)


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
        duration=16.0,
        fs=2048,
        low_frequency_cutoff=20.0,
        tukey_alpha=1.0 / 16.0,
        tfft=4.0,
        tnnw=1.0,
        height_input=256,
        kappa=1e+22
    )

    # PSD parameters
    delta_f_psd = 1.0 / sp.duration
    psdparam = {
        'length': int(sp.fs / 2 / delta_f_psd) + 1,
        'delta_f': delta_f_psd,
        'low_freq_cutoff': 5.0
    }
    psd_analytic = aLIGOZeroDetHighPower(**psdparam)

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
    # template = torch.zeros((1, ngrid_mc, sp.flen), dtype=torch.complex64)
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
        template[0, i] = torch.from_numpy(hp_fd.numpy() * sp.kappa).to(dtype=torch.complex64, device='cuda') * sp.fmask_torch
        # template[0, i] = torch.from_numpy(hp_fd.numpy()).to(dtype=torch.complex64) * sp.fmask_torch * sp.kappa
    template_conj = template.conj()
    template_2 = (template * template_conj).real

    logging.info("Generating MF images")
    tstart = time.time()
    outdir_and_label = os.path.join(outdir, label)
    if args.noise:
        for n in range(args.offset, args.offset + ndata):
            generate_noise_matchedfilter_image(outdir_and_label, n, template_conj, template_2, sp, psd_analytic)
    elif args.signal:
        psd = torch.zeros((2, 1, sp.flen))
        psd[0, 0] = torch.from_numpy(psd_analytic.numpy() * sp.kappa**2)
        psd[1, 0] = torch.from_numpy(psd_analytic.numpy() * sp.kappa**2)
        psd += 1.0e-10
        psd = psd.to(dtype=torch.float32, device='cuda')
        for n in range(args.offset, args.offset + ndata):
            generate_signal_matchedfilter_image(outdir_and_label, n, template_conj, template_2, sp, psd)
    else:
        ValueError('Something wrong with --noise and/or --signal.')
    tend = time.time()
    logging.info(f'Elapsed time: {tend - tstart} [sec]')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate matched filter images.')
    parser.add_argument('--outdir', type=str, help='Directory name including `train` or `validate` or `test`.')
    parser.add_argument('--ndata', type=int, help='Data number')
    parser.add_argument('--noise', action='store_true', help='Noise mf image')
    parser.add_argument('--signal', action='store_true', help='Signal mf image')
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # Setup logging
    if args.debug:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO
    logging.basicConfig(format='%(levelname)-8s | %(asctime)s | %(message)s',
                        level=log_level, datefmt='%Y-%m-%d %H:%M:%S')

    # Check arguments consistency
    if args.noise * args.signal:
        raise ValueError("noise and signal cannot be true at the same time.")
    elif not (args.noise or args.signal):
        raise ValueError("Specify --noise or --signal")

    main(args)
