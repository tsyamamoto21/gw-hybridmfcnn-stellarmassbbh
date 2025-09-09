import os
import re
import numpy as np
import torch
import torch.nn as nn
from pycbc.detector import Detector


class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, signal_mf_filepaths, labels, noise_mf_filepaths, transform=None):
        self.transform = transform
        self.fp_signal = signal_mf_filepaths
        self.fp_noise = noise_mf_filepaths
        self.data_num = len(signal_mf_filepaths)
        self.noise_num = len(self.fp_noise)
        self.labels = labels
        # Transforms
        self.load_zero_noise_mf = LoadZeroNoiseMatchedFilter((2, 256, 4096))
        self.proection_timeshift = ProjectionAndTimeShift()
        self.load_noise_sample = GetNoiseSample()
        self.inject = InjectSignalIntoNoise_in_MFSense()

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        zinj = self.load_zero_noise_mf(self.fp_signal[idx])
        zinj = self.proection_timeshift(zinj)
        idx_noise = np.random.randint(0, self.noise_num, (2,))
        zn = self.load_noise_sample([self.fp_noise[idx_noise[0]], self.fp_noise[idx_noise[1]]])
        out_data = self.inject(zn, zinj)
        out_label = torch.tensor(self.label[idx], dtype=torch.long)
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
            x = torch.zeros(size=self.shape, dtype=torch.complex128)
        else:
            x = torch.load(filepath, weights_only=False, dtype=torch.complex128)
        return x


class ProjectionAndTimeShift(nn.Module):
    def __init__(self):
        super().__init__()
        self.ifo1 = Detector('H1')
        self.ifo2 = Detector('L1')
        self.fs = 4096  # 4096Hz (default)
        self.w_org = 5376  # = (1.25s + 1/16s) * 4096Hz
        self.w_tar = 4096  # 4096 = 1s * 4096Hz
        self.kbuffer_for_dt = 128  # = (1/32)s * 4096Hz
        self.kbuffer_for_timeshift = 2048  # = 0.5s * 4096Hz
        self.kstart_min = self.kbuffer_for_dt
        self.kstart_max = self.kbuffer_for_dt + self.kbuffer_for_timeshift

    def forward(self, x: torch.Tensor):
        # x: torch.Tensor (2, H, W)
        _, height, width = x.size()
        # Random sampling the extrinsic parameters
        gpstime = np.random.randint(1200000000, 1400000000)
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
        kstart1 = np.random.randint(self.kstart_min, self.kstart_max)
        kend1 = kstart1 + self.w_tar
        kstart2 = kstart1 + kshift
        kend2 = kstart2 + self.w_tar
        # Crop the data
        xout = torch.zeros((2, height, self.w_tar), dtype=torch.complex128)
        xout[0] = Fp1 * Ap * x[0, :, kstart1: kend1] + Fc1 * Ac * x[1, :, kstart1: kend1]
        xout[1] = Fp2 * Ap * x[0, :, kstart2: kend2] + Fc2 * Ac * x[1, :, kstart2: kend2]
        return xout


class GetNoiseSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_org = 4 * 4096
        self.w_tar = 4096
        self.kstart_min = 0
        self.kstart_max = self.w_org - self.w_tar

    def forward(self, filepaths: list):
        channel = len(filepaths)
        xout = torch.zeros((channel, 256, 4096), dtype=torch.complex128)
        for c in range(channel):
            kstart = np.random.randint(self.kstart_min, self.kstart_max)
            xout[c] = torch.load(filepaths[c], weights_only=False)[kstart: kstart + self.w_tar]
        return xout


class InjectSignalIntoNoise_in_MFSense(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, zn: torch.Tensor, zinj: torch.Tensor):
        assert zn.size() == zinj.size(), "zn and zinj do not have the same size."
        zn2 = (zn * zn._conj()).real
        zinj2 = (zinj * zinj._conj()).real
        zcross = 2.0 * (zn * zinj._conj()).real
        return torch.sqrt(zn2 + zinj2 + zcross)


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
        target_dir = f'{datadir}/{labelname}/'
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
        target_dir = f'{datadir}/{labelname}/'
        for j in range(ndatalist[labelname]):
            input_tensors[idx] = torch.load(f"{target_dir}/{filelist[labelname][j]}")
            label_tensors[idx] = label
            print(f'{labelname}: {idx} th data loaded.')
            idx += 1
    return normalize_tensor(input_tensors), label_tensors


def make_pathlist_and_labellist(datadir, n_subset, labeldict: dict = {'noise': 0, 'cbc': 1}):
    '''
    e.g.
    datadir is the direcotry.
    labeldict = {'noise': 0, 'cbc': 1} for example

    filename pattern is 'signal_mfimage_{idx1}_{idx2}.pth'
    {idx1} is an index for a set of 1000 data.
    {idx2} is 0-999
    '''
    n_per_subset = 1000

    # List up all pth files in the direcotry
    filelist = []
    labellist = []
    # assert os.path.exists(target_dir), f"Directory `{target_dir}` does not exist."

    for k, v in labeldict.items():
        for idx_subset in range(n_subset):
            for idx in range(n_per_subset):
                if k == 'noise':
                    filename = None
                else:
                    filename = os.path.join(datadir, f'cbc/signal_mfimage_{idx_subset}_{idx}.pth')
                filelist.append(filename)
                labellist.append(v)
    return filelist, labellist


def equalize_data_number_between_labels(pathlist, labellist):
    label_unique = list(set(labellist))  # Extract unique label set
    pathsubset_list = []
    labelsubset_list = []
    ndatalist = []
    # Divide list by labels
    for label in label_unique:
        pathsubset = [pathlist[i] for i in range(len(pathlist)) if labellist[i] == label]
        pathsubset_list.append(pathsubset)
        labelsubset_list.append([label] * len(pathsubset))
        ndatalist.append(len(pathsubset))

    # Smallest label
    label_target = np.argmin(ndatalist)
    ndata_target = ndatalist[label_target]

    # Cut the dataset
    for label in label_unique:
        if label != label_target:
            pathsubset_list[label] = pathsubset_list[label][:ndata_target]
            labelsubset_list[label] = labelsubset_list[label][:ndata_target]

    pathsubset_list = sum(pathsubset_list, [])  # flatten
    labelsubset_list = sum(labelsubset_list, [])  # flatten
    return pathsubset_list, labelsubset_list
