import os
import pickle
import torch
import torch.nn as nn


class TestResult(torch.nn.Module):
    def __init__(self, result_dict):
        super(TestResult, self).__init__()
        self.label = nn.Parameter(result_dict["label"], requires_grad=False)
        self.output = nn.Parameter(result_dict["output"], requires_grad=False)


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


class SmearMFImage(nn.Module):
    def __init__(self, kfilter=16):
        super().__init__()
        self.kfilter = kfilter

    def forward(self, snrmap: torch.Tensor):
        nc, nx, ny = snrmap.shape
        ny_coarse = ny // self.kfilter
        snrmap_coarse = torch.zeros((nc, nx, ny_coarse), dtype=torch.float32)
        for i in range(ny_coarse):
            snrmap_coarse[:, :, i] = torch.sqrt(torch.mean(snrmap[:, :, i * self.kfilter: (i + 1) * self.kfilter]**2, dim=-1))
        return snrmap_coarse


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
            target_dir = os.path.join(datadir, labelname)
            assert os.path.exists(target_dir), f"Directory `{target_dir}` does not exist."
            if snr_threshold is not None:
                snrth = SNRThreshold(target_dir, idx_subset, snr_threshold)

            for idx in range(n_per_subset):
                flg = True
                if snr_threshold is not None:
                    flg = snrth.is_above_snrthreshold(idx)
                filename = os.path.join(target_dir, f'input_{idx_subset}_{idx}.pth')
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
        self.pklfile = os.path.join(dirname, f'snrlist_{idx_subset:d}.pkl')
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
