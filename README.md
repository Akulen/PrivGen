# PrivGen
RNN code used for "Privacy-Preserving Synthetic Educational Data Generation"

# Data

The data should be stored in `data/`, with the following structure:

- `data/<data>` is the folder containing the data for the dataset `<data>`
- `data/<data>/data.csv` should be the raw data, which should contain the columns `user`, `item`, `skill`, and `correct`
- `data/<data>/coef0.npy` can be generated automatically and should contain the irt coefficients matching the current dataset.
- Generated datasets should be stored in the form `data/<data>/gen-<generated-data>.csv` where `<generated-data>` is a name identifying the generation method

In the following, `<data>` and `<generated-data>` will refer to the preceding files and folders.

# Training

The following command trains a RNN model for the dataset `<data>`, and outputs the learned parameters in the file `data/<data>/params-<model_name>.pt`, where `<model_name>` contains all the model hyper-parameters, and outputs the loss curve in `data/<data>/loss-<model_name>`

```bash
python train.py <data>
```

Model hyperparameters can be adjusted, following `python train.py -h`

`bsl` is a list of integers that dictates in how many segments the sequences will be broken into during training, to allow their length to vary and work around gradient vanishing problems. Once the training has stablilized (loss hasn't improved for 100 epochs) the next element of bsl will be chosen.

# Generation

To generate a dataset with the same number of users as `<data>`, use:

```bash
python gen.py <data>
```

If model hyperparameters where adjusted during training, they must be adjusted the same way here to use the correct model parameters.

# Evaluation

## Compare IRT coefficients

For the first evaluation, the [ktm](https://github.com/jilljenn/ktm) submodule must be initiated with

```bash
git submodule init
git submodule update
```

Then, the different error values are computed with

```bash
python eval_irt.py <data> <generated-data>
```

## Evaluate Reidentification AUC

TODO: See `notebooks/Attack.ipynb` for the current code
