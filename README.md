# Hybrid MF&CNN for searching GWs from stellar mass BBHs

- paper

# Usage

## Environment

Create container
``` apptainer XXX```

## generate training data

Pure noise dataset
```
./torch_generate_mfimage.py --outdir ./data/dataset_250911/train/ --ndata 11250 --noise --offset 1250
```

Pure noise dataset
```
./torch_generate_mfimage.py --outdir ./data/dataset_250911/train/ --ndata 11250 --noise --offset 1250
```

When you generate the validation data, please replace `train` with `validate`.

## generate test data


## train the CNN


## test the CNN


## generate MDC dataset


## apply the trained CNN to MDC dataset


## Compare the sensitivities


