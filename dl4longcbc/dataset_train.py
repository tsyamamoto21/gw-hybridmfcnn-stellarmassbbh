import os
import re
import random
import numpy as np
import torch
import torch.nn as nn
from pycbc.detector import Detector


class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, signal_mf_filepaths, labels, noise_mf_filepaths, snrrange: list, smearing_kernel=None):
        # self.transform = transform
        self.fp_signal = signal_mf_filepaths
        self.fp_noise = noise_mf_filepaths
        self.data_num = len(signal_mf_filepaths)
        self.noise_num = len(self.fp_noise)
        self.labels = labels
        self.smearing_kernel = smearing_kernel
        # Transforms
        self.load_zero_noise_mf = LoadZeroNoiseMatchedFilter((2, 256, 3200))
        self.proection_timeshift = ProjectionAndTimeShift()
        self.load_noise_sample = GetNoiseSample()
        self.adjust_amplitude = AdjustAmplitudeToTargetSNR(snrrange[0], snrrange[1])
        self.inject = InjectSignalIntoNoise_in_MFSense()
        if self.smearing_kernel is not None:
            self.smear_snrmap = SmearMFImage(self.smearing_kernel)
        self.normalize = NormalizeTensor()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        # Get label
        out_label = self.labels[idx]
        # Get input data
        out_data = self.load_zero_noise_mf(self.fp_signal[idx])
        out_data = self.proection_timeshift(out_data)
        idx_noise = np.random.randint(0, self.noise_num, (2,))
        zn = self.load_noise_sample([self.fp_noise[idx_noise[0]], self.fp_noise[idx_noise[1]]])
        out_data = self.adjust_amplitude(out_data, zn, out_label)
        out_data = self.inject(zn, out_data)
        if self.smearing_kernel is not None:
            out_data = self.smear_snrmap(out_data)
        out_data = self.normalize(out_data).to(torch.float32)
        return out_data, out_label


class NormalizeTensor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # x: torch.Tensor (C, H, W)
        xmin = torch.min(x)
        xmax = torch.max(x)
        return (x - xmin) / (xmax - xmin)


class LoadZeroNoiseMatchedFilter(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, filepath):
        if filepath is None:
            x = torch.zeros(size=self.shape, dtype=torch.complex64)
        else:
            x = torch.load(filepath, weights_only=False)
            assert x.dtype == torch.complex64, "Loaded data is not torch.complex64."
            assert x.size() == self.shape, f"Size is not appropriate, loaded size {x.size()}, required {self.shape}"
        return x


class ProjectionAndTimeShift(nn.Module):
    def __init__(self):
        super().__init__()
        self.ifo1 = Detector('H1')
        self.ifo2 = Detector('L1')
        self.fs = 2048  # 2048Hz (default)
        self.w_org = 3200  # = (1.5s + 1/16s) * 2048Hz
        self.w_tar = 2048  # 2048 = 1s * 2048Hz
        self.kbuffer_for_dt = 64  # = (1/32)s * 2048Hz
        self.kbuffer_for_timeshift = 1024  # = 0.5s * 2048Hz
        self.kstart_min = self.kbuffer_for_dt
        self.kstart_max = self.kbuffer_for_dt + self.kbuffer_for_timeshift

    def forward(self, x: torch.Tensor):
        # x: torch.Tensor (2, H, W)
        _, height, width = x.size()
        # Random sampling the extrinsic parameters
        gpstime = random.randint(1200000000, 1400000000)
        ra = np.random.uniform(0.0, 2.0 * np.pi)
        dec = np.arcsin(np.random.uniform(-1, 1))
        pol = np.random.uniform(0.0, np.pi)
        cosi = np.random.uniform(-1.0, 1.0)
        # Get polarization and antenna pattern functions
        Ap = (1.0 + cosi**2) / 2.0
        Ac = cosi
        Fp1, Fc1 = self.ifo1.antenna_pattern(ra, dec, pol, gpstime)
        Fp2, Fc2 = self.ifo2.antenna_pattern(ra, dec, pol, gpstime)
        # Time Shift
        dt = self.ifo2.time_delay_from_detector(self.ifo1, ra, dec, gpstime)
        kshift = int(dt * self.fs)
        kstart1 = random.randint(self.kstart_min, self.kstart_max)
        kend1 = kstart1 + self.w_tar
        kstart2 = kstart1 + kshift
        kend2 = kstart2 + self.w_tar
        # Crop the data
        xout = torch.zeros((2, height, self.w_tar), dtype=torch.complex64)
        xout[0] = Fp1 * Ap * x[0, :, kstart1: kend1] + Fc1 * Ac * x[1, :, kstart1: kend1]
        xout[1] = Fp2 * Ap * x[0, :, kstart2: kend2] + Fc2 * Ac * x[1, :, kstart2: kend2]
        return xout


class GetNoiseSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_org = 4 * 2048
        self.w_tar = 2048
        self.kstart_min = 0
        self.kstart_max = self.w_org - self.w_tar

    def forward(self, filepaths: list):
        channel = len(filepaths)
        xout = torch.zeros((channel, 256, 2048), dtype=torch.complex64)
        for c in range(channel):
            kstart = random.randint(self.kstart_min, self.kstart_max)
            xout[c] = torch.load(filepaths[c], weights_only=False)[:, kstart: kstart + self.w_tar]
        return xout


class AdjustAmplitudeToTargetSNR(nn.Module):
    def __init__(self, snrmin, snrmax):
        super().__init__()
        self.snrmin = snrmin
        self.snrmax = snrmax

    def forward(self, zinj: torch.Tensor, zn: torch.Tensor, label):
        assert zn.size() == zinj.size(), "zn and zinj do not have the same size."
        if label == 0:
            return zinj
        elif label == 1:
            snr_target = random.randint(self.snrmin, self.snrmax)
            _, indices = torch.max(zinj.abs().view(2, -1), dim=1)
            rows, cols = torch.unravel_index(indices, (256, 2048))
            zn_max = torch.diag(zn[:, rows, cols])
            zinj_max = torch.diag(zinj[:, rows, cols])
            zn2 = torch.sum((zn_max * zn_max.conj()).real)
            zinj2 = torch.sum((zinj_max * zinj_max.conj()).real)
            zcorr = 2.0 * torch.sum((zn_max * zinj_max.conj()).real)
            amp = (-zcorr + torch.sqrt(zcorr**2 - zinj2 * (zn2 - snr_target**2))) / (zinj2)
        else:
            raise ValueError('label must be 0 or 1')
        return zinj * amp


class InjectSignalIntoNoise_in_MFSense(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, zn: torch.Tensor, zinj: torch.Tensor):
        assert zn.size() == zinj.size(), "zn and zinj do not have the same size."
        zn2 = (zn * zn._conj()).real
        zinj2 = (zinj * zinj._conj()).real
        zcross = 2.0 * (zn * zinj._conj()).real
        return torch.sqrt(zn2 + zinj2 + zcross)


class SmearMFImage(nn.Module):
    def __init__(self, kfilter=8):
        super().__init__()
        self.kfilter = kfilter

    def forward(self, snrmap: torch.Tensor):
        nc, nx, ny = snrmap.shape
        ny_coarse = ny // self.kfilter
        snrmap_coarse = torch.zeros((nc, nx, ny_coarse), dtype=torch.float32)
        for i in range(ny_coarse):
            snrmap_coarse[:, :, i] = torch.sqrt(torch.mean(snrmap[:, :, i * self.kfilter: (i + 1) * self.kfilter]**2, dim=-1))
        return snrmap_coarse


def normalize_tensor(x):
    # x.size = (N, C, H, W) is assumed.
    xmin = torch.amin(x, dim=(1, 2, 3), keepdims=True)
    xmax = torch.amax(x, dim=(1, 2, 3), keepdims=True)
    return (x - xmin) / (xmax - xmin)


def load_dataset(datadir, labelnamelist, imgsize, labellist=None):
    '''
    Assuming the file path '{datadir}/{label}/inputs_{GPSTime}_{foreground index}_{data index}.pth'.
    '''

    C, H, W = imgsize
    # The number of classes
    nclass = len(labelnamelist)
    if labellist is None:
        labellist = [i for i in range(nclass)]
    # List up all pth files in the direcotry
    filelist = {}
    pattern = re.compile(r"inputs_\d{10}_\d{2}_\d+\.pth")

    ndatalist = {}
    ndata = 0
    for labelname in labelnamelist:
        target_dir = os.path.join(datadir, labelname)
        assert os.path.exists(target_dir), f"Directory `{target_dir}` does not exist."
        all_files = os.listdir(target_dir)
        filelist[labelname] = [f for f in all_files if pattern.fullmatch(f)]
        ndatalist[labelname] = len(filelist[labelname])
        ndata += ndatalist[labelname]

    # Prepare input tensors and label tensors
    input_tensors = torch.zeros((ndata, C, H, W), dtype=torch.float32)
    label_tensors = torch.zeros((ndata,), dtype=torch.long)
    idx = 0
    for (label, labelname) in zip(labellist, labelnamelist):
        target_dir = os.path.join(datadir, labelname)
        for j in range(ndatalist[labelname]):
            input_tensors[idx] = torch.load(os.path.join(target_dir, f"{filelist[labelname][j]}"))
            label_tensors[idx] = label
            print(f'{labelname}: {idx} th data loaded.')
            idx += 1
    return normalize_tensor(input_tensors), label_tensors


def make_pathlist_and_labellist(datadir: str, ndata: int, labeldict: dict = {'noise': 0, 'cbc': 1}):
    '''
    e.g.
    datadir is the direcotry.
    labeldict = {'noise': 0, 'cbc': 1} for example

    filename pattern is 'signal_mfimage_{idx}.pth'
    '''
    # List up all pth files in the direcotry
    filelist = []
    labellist = []
    # assert os.path.exists(target_dir), f"Directory `{target_dir}` does not exist."

    for k, v in labeldict.items():
        for idx in range(ndata):
            if k == 'noise':
                filename = None
            else:
                filename = os.path.join(datadir, f'cbc/signalmf_{idx:d}.pth')
                assert os.path.exists(filename), f"File {filename} does not exist."
            filelist.append(filename)
            labellist.append(v)
    return filelist, torch.tensor(labellist, dtype=torch.long)


def get_noise_filepaths(datadir: str, nfile: int):
    filelist = []
    for idx in range(nfile):
        filename = os.path.join(datadir, f'noise/noisemf_{idx:d}.pth')
        assert os.path.exists(filename), f"File {filename} does not exist."
        filelist.append(filename)
    return filelist


def make_snrmap_coarse(snrmap, kfilter):
    nc, nx, ny = snrmap.shape
    ny_coarse = ny // kfilter
    snrmap_coarse = torch.zeros((nc, nx, ny_coarse), dtype=torch.float32)
    for i in range(ny_coarse):
        snrmap_coarse[:, :, i] = torch.sqrt(torch.mean(snrmap[:, :, i * kfilter: (i + 1) * kfilter]**2, dim=-1))
    return snrmap_coarse
