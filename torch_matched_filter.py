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
        self.flen = self.tlen // 2 + 1

        # FFT for PSD estimation
        self.tfft = tfft
        self.toverlap = self.tfft / 2
        self.fftlength = int(self.tfft * self.fs)
        self.overlaplength = int(self.fftlength / 2)

        # MF params
        self.tukey_alpha = tukey_alpha
        self.window = tukey(self.tlen, self.tukey_alpha)
        self.window_jax = jnp.array(self.window)

        # Image properties
        self.width_input = width_input
        self.height_input = height_input

        # Get frequency mask
        self.fmask = self._get_frequency_mask()

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
    assert h2_mat.ndim == 2, "h2_mat does not have dim of 2."
    assert h2_mat.shape[1] == len(psd), "template in frequency domain and PSD does not have the same length."
    return 4.0 * np.trapezoid(h2_mat / psd, dx=sp.df, axis=1)


def numpy_matched_filter(strain_td: np.ndarray, hconj_mat: np.ndarray, psd: np.ndarray, norm: np.ndarray, sp: SignalProcessingParameters) -> np.ndarray:
    '''
    * hconj must be multiplied by fmask
    * norm = sigmasq = (h|h)
    '''
    assert hconj_mat.ndim == 2, "hp2_mat does not have dim of 2."
    assert hconj_mat.shape[1] == len(psd), "template in frequency domain and PSD does not have the same length."
    assert hconj_mat.shape[0] == len(norm), "templates in frequency domain and norm does not have the same length."

    strain_fd = np.fft.rfft(strain_td) * sp.dt
    return 4.0 * np.fft.ifft(strain_fd * hconj_mat / psd, sp.tlen, axis=1) / sp.dt / np.sqrt(norm[:, np.newaxis])


# normalize template
def torch_sigmasq(h2_mat: torch.Tensor, psd: torch.Tensor, df: float) -> float:
    assert h2_mat.ndim == 2, "h2_mat does not have dim of 2."
    assert h2_mat.shape[1] == len(psd), "template in frequency domain and PSD does not have the same length."
    return 4.0 * torch.trapezoid(h2_mat / psd, dx=df, dim=-1)


def torch_matched_filter(strain_td: torch.Tensor, hconj_mat: torch.Tensor, psd: torch.Tensor, norm: torch.Tensor, tlen: int, dt: float) -> torch.Tensor:
    '''
    * hconj must be multiplied by fmask
    * norm = sigmasq = (h|h)
    '''
    assert hconj_mat.ndim == 3, "hp2_mat does not have dim of 2."
    assert hconj_mat.shape[0] == len(norm), "templates in frequency domain and norm does not have the same length."
    assert hconj_mat.shape[2] == len(psd), "template in frequency domain and PSD does not have the same length."

    strain_fd = jnp.fft.rfft(strain_td) * dt
    return 4.0 * jnp.fft.ifft(strain_fd * hconj_mat / psd, tlen, axis=1) / dt / jnp.sqrt(norm[:, jnp.newaxis])


# Computational speed
# Generate signal to be processed
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

noise = noise_from_string('aLIGOZeroDetHighPower', length=sp.tlen, delta_t=sp.dt, low_frequency_cutoff=sp.low_frequency_cutoff)
noise_td_np = noise.numpy()
noise_td_jax = jnp.array(noise_td_np)

# Estimate PSD
psd_original = noise.psd(sp.tfft, avg_method='median-mean')
psd_pycbc = interpolate(psd_original, sp.df, sp.flen)
psd = psd_pycbc.numpy()
psd_jax = jnp.array(psd)

ntemplist = 2 ** np.arange(2, 10)
nptimelist = []
jaxtimelist = []
pycbctimelist = []
for ntemp in ntemplist:
    # template bank
    mgrid = np.linspace(5, 100, ntemp, endpoint=True)
    template = np.zeros((ntemp, sp.flen), dtype=np.complex128)
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

        template_pycbc.append(hp_pycbc)
        template[n] = hp_pycbc.numpy() * sp.fmask
    template_conj = template.conjugate()
    template_2 = (template * template_conj).real

    template_conj_jax = jnp.array(template_conj)
    template_2_jax = jnp.array(template_2)

    # Matched filter by numpy
    tik = time.time()
    sigmasq_np = numpy_sigmasq(template_2, psd, sp)
    matched_filter_np = numpy_matched_filter(noise_td_np * sp.window, template_conj, psd, sigmasq_np, sp)
    tok = time.time()
    nptimelist.append(tok - tik)

    # Matched filter by jax
    tik = time.time()
    sigmasq_np = jax_sigmasq(template_2_jax, psd_jax, sp.df)
    matched_filter_jax = jax_matched_filter(noise_td_jax * sp.window, template_conj_jax, psd_jax, sigmasq_np, sp.tlen, sp.dt)
    tok = time.time()
    jaxtimelist.append(tok - tik)

    # matched filter by PyCBC
    tik = time.time()
    matched_filter_pycbc = []
    for n in range(ntemp):
        matched_filter_pycbc.append(matched_filter(template_pycbc[n], noise * sp.window, psd_pycbc, sp.low_frequency_cutoff, sp.high_frequency_cutoff))
    matched_filter_pycbc = np.array(matched_filter_pycbc)
    tok = time.time()
    pycbctimelist.append(tok - tik)


plt.figure()
plt.loglog(ntemplist, nptimelist, "s-", label="numpy")
plt.loglog(ntemplist, jaxtimelist, "o-", label="jax")
plt.loglog(ntemplist, pycbctimelist, "p-", label="pycbc")
plt.xlabel('Number of templates')
plt.ylabel('Elapsed time [s]')
plt.legend()
plt.grid()

plt.figure()
plt.plot(ntemplist, np.divide(pycbctimelist, nptimelist), "s-", label="numpy")
plt.plot(ntemplist, np.divide(pycbctimelist, jaxtimelist), "o-", label="jax")
plt.xscale('log')
plt.xlabel('Number of templates')
plt.ylabel('Speed up factor')
plt.legend()
plt.grid()

plt.show()
