import os
import re
import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from pycbc.detector import Detector


class TestResult(torch.nn.Module):
    def __init__(self, result_dict):
        super(TestResult, self).__init__()
        self.label = nn.Parameter(result_dict["label"], requires_grad=False)
        self.output = nn.Parameter(result_dict["output"], requires_grad=False)


class MyDataset(Dataset):
    def __init__(self, data, target, transform=None):
        self.data = data
        self.target = target
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.data[index]
        if self.transform:
            x = self.transform(x)
        y = self.target[index]
        return x, y


class LabelDataset(torch.utils.data.Dataset):
    '''
    data: List of paths
    label: List of labels
    '''
    def __init__(self, data, label, transform=None):
        self.transform = transform
        self.data = data
        self.data_num = len(data)
        self.label = label

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_data = self.transform(torch.load(self.data[idx], weights_only=True))
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
            x = torch.zeros(size=self.shape, dtype=torch.float32)
        else:
            x = torch.load(filepath, weights_only=False, dtype=torch.float32)
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
        self.kstart_min = self.k_buffer_for_dt
        self.kstart_max = self.k_buffer_for_dt + self.k_buffer_for_timeshift

    def forward(self, x):
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
        xout = torch.zeros((2, height, self.w_tar))
        xout[0] = Fp1 * Ap * x[0, :, kstart1: kend1] + Fc1 * Ac * x[1, :, kstart1: kend1]
        xout[1] = Fp2 * Ap * x[0, :, kstart2: kend2] + Fc2 * Ac * x[1, :, kstart2: kend2]
        return xout


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


def make_pathlist_and_labellist(datadir, n_subset, labelnames, labels=None, snr_threshold=None):
    '''
    e.g.
    datadir is the direcotry.
    labelnames = ['noise', 'cbc'] or ['noise', 'cbc', 'glitch']
    labels = [0, 1] or [0, 1, 2]

    filename pattern is 'input_{idx1}_{idx2}.pth'
    {idx1} is an index for a set of 1000 data.
    {idx2} is 0-999
    '''

    n_per_subset = 1000
    # The number of classes
    nclass = len(labelnames)
    if labels is None:
        labels = [i for i in range(nclass)]

    # List up all pth files in the direcotry
    filelist = []
    labellist = []
    for label, labelname in zip(labels, labelnames):
        for idx_subset in range(n_subset):
                target_dir = f'{datadir}/{labelname}/'
                assert os.path.exists(target_dir), f"Directory `{target_dir}` does not exist."
                if snr_threshold is not None:
                    snrth = SNRThreshold(target_dir, idx_subset, snr_threshold)

                for idx in range(n_per_subset):
                    flg = True
                    if snr_threshold is not None:
                        flg = snrth.is_above_snrthreshold(idx)
                    filename = f'{target_dir}/input_{idx_subset}_{idx}.pth'
                    if os.path.exists(filename) and flg:
                        filelist.append(filename)
                        labellist.append(label)
    return filelist, labellist


class SNRThreshold():
    def __init__(self, dirname, idx_subset, snr_threshold, mode='either'):
        '''
        dirname should include up to label
        '''
        assert (mode == 'both') or (mode == 'either'), 'Choose both or either'
        self.dirname = dirname
        self.pklfile = f'{dirname}/snrlist_{idx_subset:d}.pkl'
        with open(self.pklfile, 'rb') as fo:
            self.snrlist = pickle.load(fo)
        self.snr_threshold = snr_threshold
        self.mode = mode

    def is_above_snrthreshold(self, idx_data):
        flg_above_threshold_h = self.snrlist[idx_data][2] >= self.snr_threshold
        flg_above_threshold_l = self.snrlist[idx_data][3] >= self.snr_threshold
        if self.mode == 'both':
            flg_above_threshold = flg_above_threshold_h and flg_above_threshold_l
        elif self.mode == 'either':
            flg_above_threshold = flg_above_threshold_h or flg_above_threshold_l
        return flg_above_threshold


def equalize_data_number_between_labels(pathlist, labellist):
    label_unique = list(set(labellist))  # Extract unique label set
    pathsubset_list = []
    labelsubset_list = []
    ndatalist = []
    # Divide list by labels
    for l in label_unique:
        pathsubset = [pathlist[i] for i in range(len(pathlist)) if labellist[i] == l]
        pathsubset_list.append(pathsubset)
        labelsubset_list.append([l]*len(pathsubset))
        ndatalist.append(len(pathsubset))

    # Smallest label
    label_target = np.argmin(ndatalist)
    ndata_target = ndatalist[label_target]

    # Cut the dataset
    for l in label_unique:
        if l != label_target:
            pathsubset_list[l] = pathsubset_list[l][:ndata_target]
            labelsubset_list[l] = labelsubset_list[l][:ndata_target]
    
    pathsubset_list = sum(pathsubset_list, [])  # flatten
    labelsubset_list = sum(labelsubset_list, [])  # flatten
    return pathsubset_list, labelsubset_list