"""
utils.py
"""
import os
import numpy as np
from pycbc.waveform import TimeSeries
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from .gw_parameters import remnantspin, ISCO_radius
from astropy.constants import G, c, M_sun, pc
GRAV_CONST = G.value
SPEED_OF_LIGHT = c.value
M_SUN = M_sun.value
MPC = pc.value * 1.0e+6


def if_not_exist_makedir(dirname):
    if os.path.exists(dirname):
        pass
    else:
        os.makedirs(dirname)


def get_chirpmass(m1, m2):
    return ((m1 * m2)**(3. / 5.)) / ((m1 + m2)**(1. / 5.))


def get_dfdt(Mc, f):
    prefactor = 96.0 * (np.pi**(8.0 / 3.0)) / 5.0
    numfactor = (GRAV_CONST * M_SUN / (SPEED_OF_LIGHT ** 3.0)) ** (5.0 / 3.0)
    return f**(11. / 3.) * Mc**(5. / 3.) * prefactor * numfactor


def startfrequency(Mc, tobs, fmerge):
    factor = 256.0 * (np.pi**(8. / 3.)) / 5.0
    massfactor = (GRAV_CONST * Mc * M_SUN / (SPEED_OF_LIGHT**3.0))**(5.0 / 3.0)
    return (fmerge**(-8.0 / 3.0) + tobs * factor * massfactor) ** (-3.0 / 8.0)


def get_isco_frequency(m1, m2):
    """Calculate ISCO frequency

    Args:
        m1 (float): [M_sun] Primary mass
        m2 (float): [M_sun] Secondary mass

    Returns:
        float: [Hz] ISCO frequency
    """
    mtot = m1 + m2
    arem = remnantspin(m1, m2, 0.0, 0.0)
    Risco = ISCO_radius(arem) * GRAV_CONST * M_SUN * mtot / (SPEED_OF_LIGHT**2)
    fisco = np.sqrt(GRAV_CONST * M_SUN * mtot / (Risco**3)) / (np.pi)
    return fisco


def expectedsnr(freq, psd, Mc, d, z, Ndetector=1):
    # distance in Mpc
    # Mc in M_sun
    # Eq.(16) of Dalal et al. astro-ph/0601275
    amp = np.sqrt(5.0 / 96.0) * SPEED_OF_LIGHT * (GRAV_CONST * (1 + z) * Mc * M_SUN / (SPEED_OF_LIGHT**3.0))**(5. / 6.) / (np.pi**(2. / 3.))
    # integrate = np.trapz(((1+z) * freq) ** (-7.0/3.0) / psd, freq)
    integrate = np.trapz((freq ** (-7.0 / 3.0)) / psd, freq)
    return 1.6 * amp * np.sqrt(integrate * Ndetector) / (d * MPC)


def adjust_waveform_length(hp: TimeSeries, hc: TimeSeries, tsignal: float, merger: float = 0.9):
    tstart = hp.sample_times[0]
    tend = hp.sample_times[-1]
    fs = hp.sample_rate
    dt_start = abs(tstart) - (tsignal * merger)
    dt_end = tend - (tsignal * (1.0 - merger))
    # Adjusting the signal length
    if dt_start >= 0.0:
        hp = hp.crop(dt_start, 0.0)
        hc = hc.crop(dt_start, 0.0)
    else:
        hp.prepend_zeros(int(abs(dt_start) * fs))
        hc.prepend_zeros(int(abs(dt_start) * fs))
    if dt_end >= 0.0:
        hp = hp.crop(0.0, dt_end)
        hc = hc.crop(0.0, dt_end)
    else:
        hp.append_zeros(int(abs(dt_end) * fs))
        hc.append_zeros(int(abs(dt_end) * fs))
    if len(hp) != int(tsignal * fs):
        hp = hp.crop(1.0 / fs, 0.0)
    if len(hc) != int(tsignal * fs):
        hc = hc.crop(1.0 / fs, 0.0)
    return hp, hc


def plot_training_curve(trainloss, validateloss, filename, xlabel='Epoch', ylabel='Loss'):
    tr_epoch_list = [row[2] for row in trainloss]
    trloss = [row[1] for row in trainloss]
    val_epoch_list = [row[2] for row in validateloss]
    valloss = [row[1] for row in validateloss]
    plt.figure()
    plt.plot(tr_epoch_list, trloss, label='train')
    plt.plot(val_epoch_list, valloss, label='validate')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.savefig(filename)


class SNRRangeScheduler:
    def __init__(self, config: OmegaConf):
        self.previous_step = None
        self.config = config
        self.schedule = config.train.snr_schedule
        self.nstep = len(self.schedule)
        self._check_if_snr_schdule_is_well_configured(config)

    def step(self, epoch):
        for i in range(self.nstep):
            if self.schedule[i][0] <= epoch < self.schedule[i][1]:
                current_step = i

        snrrange = [self.schedule[current_step][2], self.schedule[current_step][3]]
        flag = (current_step != self.previous_step)
        self.previous_step = current_step
        return flag, snrrange

    def _check_if_snr_schdule_is_well_configured(self, config: OmegaConf):
        schedule = config.train.snr_schedule
        nsteps = len(schedule)

        # Size check
        length_is_not_4 = np.any([len(schedule[i]) != 4 for i in range(nsteps)])
        if length_is_not_4:
            raise ValueError(f"Some schedule step has no list with the length of 4: Schedule = {schedule}")

        # The first epoch should be 0
        if schedule[0][0] != 0:
            raise ValueError(f"The first epoch of the first step should be 0: Schedule = {schedule}")

        # The last epoch should equal the num_epochs
        if schedule[nsteps - 1][1] != config.train.num_epochs:
            raise ValueError(f"The last epoch of the last step should be num_epochs: Schedule = {schedule}")

        # Consistency check
        errorflg = False
        if nsteps >= 2:
            for i in range(nsteps - 1):
                errorflg += not (schedule[i][1] == schedule[i + 1][0])
            if errorflg:
                raise ValueError(f"Step start epoch does not match the end epoch of the previous step: Schedule = {schedule}")

        # SNR max must larger than SNR min
        errorflg = False
        for i in range(nsteps):
            errorflg += schedule[i][2] > schedule[i][3]
        if errorflg:
            raise ValueError(f"SNR max is smaller than SNR min: Schedule = {schedule}")
