import argparse
import numpy as np
import pandas as pd
import torch

from models import RNN


def make_parser(desc):
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--use_gpu',
                        action='store_true',
                        default=False,
                        help='Use the gpu (requires CUDA)')
    return parser


def add_dataset_args(parser):
    parser.add_argument('data',
                        type=str,
                        help='Dataset to use')


def add_model_params(parser):
    parser.add_argument('--hidden_dim',
                        type=int,
                        default=200,
                        help='Hidden dimension of the RNN')
    parser.add_argument('--n_layers',
                        type=int,
                        default=2,
                        help='Number of layers of the RNN')
    parser.add_argument('--n_epochs',
                        type=int,
                        default=10000,
                        help='Number of epochs')
    parser.add_argument('--lr',
                        type=float,
                        default=0.001,
                        help='Learning rate')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help='Size of the batches')
    parser.add_argument('--embed_size',
                        type=int,
                        default=16,
                        help='Size of the skill embeddings')
    parser.add_argument('--predict_outcome',
                        action='store_true',
                        default=False,
                        help='Train the RNN to predict the outcome')
    parser.add_argument('--balance',
                        action='store_true',
                        default=False,
                        help='Weight the loss to compensate for the skills with less samples')
    parser.add_argument('--bsl',
                        nargs='+',
                        type=int,
                        default=[20, 10, 5, 5],
                        help='List of sequences lengths during training')
    parser.add_argument('--start',
                        type=str,
                        default=None,
                        help='Start the training with the parameters stored in START')


def get_device(options):
    return torch.device("cuda") if options.use_gpu else torch.device("cpu")


def load_data(options, user='user'):
    df = pd.read_csv(f'data/{options.data}/data.csv')

    df["skill_id"] = df["skill"].astype('category')
    df["skill_id"] = df["skill_id"].cat.codes
    n_items = None  # df['item'].nunique()
    n_skills = df['skill_id'].nunique()
    N_SKILLS = df['skill'].max() + 1
    # n_items = df['item_id'].nunique()
    # items_per_skill = df.groupby('skill_id')['item_id'].nunique()

    n_user = df[user].nunique()
    coef = np.load(f'data/{options.data}/coef0.npy')

    assert(len(coef[0]) - (df['user'].max() + 1) - N_SKILLS == 0)

    theta, e = coef[0, :-N_SKILLS], coef[0, -N_SKILLS:]

    print('coef read')

    return df, (n_user, n_skills, n_items), (theta, e)


def make_RNN(n_skills, n_items, device, options):
    model_name = f'{options.bsl}-{options.hidden_dim}-{options.n_layers}-' \
                 + f'{options.embed_size}-{options.lr}-' \
                 + f'{int(options.predict_outcome)}-{options.batch_size}-' \
                 + f'{int(options.balance)}'
    rnn = RNN(n_skills, n_items, -1, hidden_dim=[options.hidden_dim],
              n_layers=[options.n_layers], embed_size=options.embed_size,
              n_epochs=options.n_epochs, lr=options.lr,
              predict_outcome=options.predict_outcome,
              batch_size=options.batch_size, bsl=options.bsl,
              balance=options.balance, device=device)
    return model_name, rnn
