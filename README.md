# Hybrid MF&CNN for searching GWs from stellar mass BBHs

- paper

# Usage

## Environment

Create container
``` apptainer XXX```

## generate training data

Pure noise dataset
```
./torch_generate_mfimage.py --outdir <path_to_dataset_directory>/train/ --ndata 1250 --noise
```

Pure CBC dataset
```
./torch_generate_mfimage.py --outdir <path_to_dataset_directory>/train/ --ndata 10000 --signal
```

When you generate the validation data, please replace `train` with `validate`.

## generate test data

Noise dataset
```
./generate_matched_filter_image.py --outdir <path_to_dataset_directory>/test/ --ndata 10000 --config config/dataset.ini --starttime 1284169603 --noiseonly
```

CBC dataset
```
./generate_matched_filter_image.py --outdir <path_to_dataset_directory>/test/ --ndata 10000 --config config/dataset.ini --starttime 1284169603
```

## train the CNN

Before running `./train.py`, the training configuration file must be prepared at `./config/config_train.yaml`. 

```
./train.py --dirname <directory_name>
```

You can change the path to the config file. In that case, you specify the option `--config <path to config file>`.

The trained model and its relevant information files are stored at the directory `./data/model/<experiment>/<directory_name>`. Here, `<experiment>` is specified in the configuration file.

## test the CNN

Test with noise dataset
```
./test.py --outdir=<path_to_cnn_model>/test_noise/ --modeldir=<path_to_cnn_model> --datadir=<path_to_dataset_directory>/test/ --ndata=10000 --batchsize=100 --noise
```

Test with CBC dataset
```
./test.py --outdir=<path_to_cnn_model>/test_cbc/ --modeldir=<path_to_cnn_model> --datadir=<path_to_dataset_directory>/test/ --ndata=10000 --batchsize=100 --cbc
```

Notebook `XXX` is used for analyzing the test results.


## generate MDC dataset

```
./mdc/generate_data.py\
	-d ds1\
    -i <mdc_data_directory>/injection.hdf\
    -f <mdc_data_directory>/foreground.hdf\
    -b <mdc_data_directory>/background.hdf\
	-s 2514409456\
	--duration 2592000\
	--verbose
```

## apply the trained CNN to MDC dataset

Process background data

```
./mdc_main.py\
   -i <mdc_data_directory>/background.hdf\
   -o <path_to_cnn_model>/ds1/bg.hdf\
   --modeldir <path_to_cnn_model>
```

Process foreground data

```
./mdc_main.py\
   -i <mdc_data_directory>/foreground.hdf\
   -o <path_to_cnn_model>/ds1/fg.hdf\
   --modeldir <path_to_cnn_model>
```

Evaluate the results

```
./mdc/evaluate.py\
    --injection-file <mdc_data_directory>/injection.hdf\
    --foreground-events <path_to_cnn_model>/ds1/fg.hdf\
    --foreground-files <mdc_data_directory>/foreground.hdf\
    --background-events <path_to_cnn_model>/ds1/bg.hdf\
    --output-file <path_to_cnn_model>/ds1/eval.hdf\
```


## Compare the sensitivities

Run the notebook `XXX`
