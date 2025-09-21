import os
# import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey
from pycbc.psd import interpolate
from pycbc.noise import noise_from_string
from pycbc.waveform import get_fd_waveform
from pycbc.filter import matched_filter


class SignalProcessingParameters:
    def __init__(self, duration, tsegment, fs, low_frequency_cutoff, tfft, tukey_alpha, width_input, height_input, kappa):
        # Signal properties
        self.duration = duration
        self.tsegment = tsegment
        self.fs = fs
        self.low_frequency_cutoff = low_frequency_cutoff
        self.high_frequency_cutoff = self.fs / 2
        self.dt = 1.0 / self.fs
        self.df = 1.0 / self.duration
        self.tlen = int(self.duration * self.fs)
        self.flen = self.tlen // 2 + 1

        # FFT for PSD estimation
        self.tfft = tfft
        self.toverlap = self.tfft / 2
        self.fftlength = int(self.tfft * self.fs)
        self.overlaplength = int(self.fftlength / 2)

        # MF params
        self.kappa = kappa
        self.tukey_alpha = tukey_alpha
        self.window = tukey(self.tlen, self.tukey_alpha)
        self.window_torch = torch.from_numpy(self.window).to(torch.float32).to(device='cuda')

        # Image properties
        self.width_input = width_input
        self.height_input = height_input
        # self.kfilter = int(self.fs * self.tsegment / self.width_input)
        # self.kcrop_left = int(self.fs * (self.duration / 2 - 3 * self.tsegment / 4))
        # self.kcrop_right = int(self.fs * (self.duration / 2 + 3 * self.tsegment / 4))
        # self.width_before_smearing = self.kcrop_right - self.kcrop_left

        # Get frequency mask
        self.fmask = self._get_frequency_mask()
        self.fmask_torch = torch.from_numpy(self.fmask).to(torch.float32).to('cuda')

    def _get_frequency_mask(self) -> np.ndarray:
        # Get mask
        fmask = np.zeros((self.flen))
        kmin = int(self.low_frequency_cutoff / self.df)
        kmax = int(self.high_frequency_cutoff / self.df)
        if kmax > self.flen:
            print(f"kmax ({kmax}) is larger than flen ({self.flen}).")
        fmask[kmin: kmax] = 1.0
        return fmask


# normalize template
def numpy_sigmasq(h2_mat: np.ndarray, psd: np.ndarray, sp: SignalProcessingParameters):
    assert h2_mat.ndim == 3, "h2_mat does not have dim of 3."
    assert h2_mat.shape[0] == 1, f"shape error: {h2_mat.shape=}"
    assert psd.ndim == 3, f"shape error: {psd.shape=}"
    assert psd.shape[0] == 2, f"shape error: {psd.shape=}"
    assert psd.shape[1] == 1, f"shape error: {psd.shape=}"
    assert h2_mat.shape[2] == psd.shape[2], "template in frequency domain and PSD does not have the same length."

    ntemp = h2_mat.shape[1]
    return 4.0 * np.trapezoid(h2_mat / psd, dx=sp.df, axis=-1).reshape(2, ntemp, 1)


def numpy_matched_filter(strain_td: np.ndarray, hconj_mat: np.ndarray, psd: np.ndarray, norm: np.ndarray, sp: SignalProcessingParameters) -> np.ndarray:
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

    strain_fd = np.fft.rfft(strain_td * sp.window, axis=-1) * sp.dt
    return 4.0 * np.fft.ifft(strain_fd * hconj_mat / psd, sp.tlen, axis=-1) / sp.dt / np.sqrt(norm)


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


# Computational speed
outdir = 'data/torch_matched_filter'
if not os.path.exists(outdir):
    os.makedirs(outdir)

# Generate signal to be processed
sp = SignalProcessingParameters(
    duration=16.0,
    tsegment=1.0,
    fs=2048,
    low_frequency_cutoff=20.0,
    tukey_alpha=1.0 / 16.0,
    tfft=4.0,
    width_input=2048,
    height_input=256,
    kappa=1e+22
)

noise_h = noise_from_string('aLIGOZeroDetHighPower', length=sp.tlen, delta_t=sp.dt, low_frequency_cutoff=sp.low_frequency_cutoff)
noise_l = noise_from_string('aLIGOZeroDetHighPower', length=sp.tlen, delta_t=sp.dt, low_frequency_cutoff=sp.low_frequency_cutoff)
# To numpy
noise_td_np = np.zeros((2, 1, sp.tlen))
noise_td_np[0] = noise_h.numpy()
noise_td_np[1] = noise_l.numpy()
# To torch
noise_td_torch = torch.zeros((2, 1, sp.tlen), device='cuda')
noise_td_torch[0] = torch.from_numpy(noise_h.numpy() * sp.kappa).to(torch.float32).to('cuda')
noise_td_torch[1] = torch.from_numpy(noise_l.numpy() * sp.kappa).to(torch.float32).to('cuda')

# Estimate PSD
psdh_original = noise_h.psd(sp.tfft, avg_method='median-mean')
psdl_original = noise_l.psd(sp.tfft, avg_method='median-mean')
psdh_pycbc = interpolate(psdh_original, sp.df, sp.flen)
psdl_pycbc = interpolate(psdl_original, sp.df, sp.flen)
# to numpy
psd_np = np.zeros((2, 1, sp.flen))
psd_np[0] = psdh_pycbc.numpy()
psd_np[1] = psdl_pycbc.numpy()
# to torch
psd_torch = torch.zeros((2, 1, sp.flen), dtype=torch.float32, device='cuda')
psd_torch[0] = torch.from_numpy(psdh_pycbc.numpy() * sp.kappa**2).to(torch.float32).to('cuda')
psd_torch[1] = torch.from_numpy(psdl_pycbc.numpy() * sp.kappa**2).to(torch.float32).to('cuda')

nptimelist = []
torchtimelist = []
pycbctimelist = []
errlist = []
ntemplist = 2 ** np.arange(2, 10)
for ntemp in ntemplist:
    # template bank
    mgrid = np.linspace(5, 50, ntemp, endpoint=True)
    template_np = np.zeros((1, ntemp, sp.flen), dtype=np.complex128)
    template_torch = torch.zeros((1, ntemp, sp.flen)).to(torch.complex64).to('cuda')
    template_pycbc = []

    for n in range(ntemp):
        hp_pycbc, hc_pycbc = get_fd_waveform(
            approximant='IMRPhenomD',
            mass1=mgrid[n],
            mass2=mgrid[n],
            delta_f=sp.df,
            f_lower=sp.low_frequency_cutoff,
            f_final=sp.high_frequency_cutoff
        )

        template_pycbc.append(hp_pycbc * sp.kappa)
        template_np[0, n] = hp_pycbc.numpy() * sp.fmask
        template_torch[0, n] = torch.from_numpy(hp_pycbc.numpy()).to(torch.complex64).to('cuda') * sp.fmask_torch * sp.kappa
        # hp_conj = hp.conjugate()
        # hp2 = (hp * hp_conj).real
    template_conj_np = template_np.conjugate()
    template_2_np = (template_np * template_conj_np).real
    template_conj_torch = template_torch.conj()
    template_2_torch = (template_torch * template_conj_torch).real

    # Matched filter by numpy
    tik = time.time()
    sigmasq_np = numpy_sigmasq(template_2_np, psd_np, sp)
    matched_filter_np = numpy_matched_filter(noise_td_np, template_conj_np, psd_np, sigmasq_np, sp)
    tok = time.time()
    nptimelist.append(tok - tik)

    # Matched filter by torch
    tik = time.time()
    sigmasq_torch = torch_sigmasq(template_2_torch, psd_torch, sp)
    matched_filter_torch = torch_matched_filter(noise_td_torch, template_conj_torch, psd_torch, sigmasq_torch, sp)
    tok = time.time()
    torchtimelist.append(tok - tik)

    # Matched filter by Pycbc
    tik = time.time()
    matched_filter_pycbc_h = []
    matched_filter_pycbc_l = []
    for n in range(ntemp):
        matched_filter_pycbc_h.append(matched_filter(template_pycbc[n], noise_h * sp.window, psdh_pycbc, sp.low_frequency_cutoff, sp.high_frequency_cutoff))
        matched_filter_pycbc_l.append(matched_filter(template_pycbc[n], noise_l * sp.window, psdl_pycbc, sp.low_frequency_cutoff, sp.high_frequency_cutoff))
    tok = time.time()
    pycbctimelist.append(tok - tik)

    err = np.max(abs(matched_filter_torch[0].cpu().numpy() - np.array(matched_filter_pycbc_h)))
    errlist.append(err)


# Check the matched filter results
n = 0

fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, tight_layout=True)
ax[0].plot(matched_filter_pycbc_h[n].real(), label='pycbc', lw=3, color='k', alpha=0.4)
ax[0].plot(matched_filter_np[0, n].real, label='numpy', lw=1)
ax[0].plot(matched_filter_torch[0, n].real.cpu(), label='torch', linestyle='--', lw=1)
ax[1].plot(matched_filter_pycbc_h[n].imag(), label='pycbc', lw=3, color='k', alpha=0.4)
ax[1].plot(matched_filter_np[0, n].imag, label='numpy', lw=1)
ax[1].plot(matched_filter_torch[0, n].imag.cpu(), label='torch', linestyle='--', lw=1)
ax[0].set(ylabel='SNR time series', xlim=[5 * sp.fs, 5.3 * sp.fs])
ax[1].set(xlabel='time [s]', ylabel='SNR time series', xlim=[5 * sp.fs, 5.3 * sp.fs])
ax[0].legend()
fig.savefig(os.path.join(outdir, 'snr_time_series_comparison.png'))

plt.figure()
plt.plot(errlist, "s", c="k")
plt.xlabel('Mass (m1=m2)')
plt.ylabel('Max of relative error in SNR')
plt.grid()
plt.savefig(os.path.join(outdir, 'relative_erros_in_snr.png'))

fig, ax = plt.subplots(2, 1, figsize=(8, 8), sharex=True, tight_layout=True)
ax[0].loglog(ntemplist, pycbctimelist, "p-", label="pycbc", c='k')
ax[0].loglog(ntemplist, nptimelist, "s-", label="numpy")
ax[0].loglog(ntemplist, torchtimelist, "o-", label="torch")
ax[1].plot(ntemplist, np.divide(pycbctimelist, nptimelist), "s-", label="numpy")
ax[1].plot(ntemplist, np.divide(pycbctimelist, torchtimelist), "o-", label="torch")
ax[0].set(ylabel='Elapsed time [s]')
ax[0].legend()
ax[0].grid()
ax[1].set(xlabel='Number of templates', ylabel='Speed up factor', xscale='log')
ax[1].legend()
ax[1].grid()
fig.savefig(os.path.join(outdir, 'computationa_time_and_speed_up_factor.png'))
