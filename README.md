# PrivGen
RNN code used for "Privacy-Preserving Synthetic Educational Data Generation"

# Evaluation

## Compare IRT coefficients

For the first evaluation, the [ktm](https://github.com/jilljenn/ktm) submodule must be initiated with

```bash
git submodule init
git submodule update
```

Then, the different error values are computed with

```bash
python prepare_data.py <data> <generated-data>
```

where data is a folder in `data`, containing the training data `data/<data>/data.csv`, the corresponding ktm coefficients `data/<data>/coef0.npy` and the generated data `data/<data>/gen-<generated-data>.csv`
