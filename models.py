import logging
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tqdm import trange
from torch import nn
import numpy as np
from sklearn.utils import gen_batches

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def irt(theta, e):
    return sigmoid(theta + e)

class RNN(nn.Module):
    def __init__(self, n_skills, n_items, items_per_skill, hidden_dim=[20, 20],
            n_layers=[1, 1], embed_size=16, n_epochs=100, lr=0.01,
            predict_outcome=False, batch_size=None, bsl=None, balance=False,
            device=None):
        super().__init__()
        self.logger = logging.getLogger(__name__
                                      + '.'
                                      + self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.n_epochs = n_epochs
        # Learning rate
        self.lr = lr
        self.batch_size = batch_size
        self.n_skills = n_skills
        self.predict_outcome = predict_outcome
        # List of sequence lengths to use for the training.
        # Once the training has stabilized, move on to the next one
        self.bsl = bsl
        # Weight the loss to reduce the importance of common skills
        self.balance = balance
        # TODO: pass on GPU if available and batch_size allows for it
        self.device = torch.device("cpu") if device is None else device

        self.embed = nn.Embedding(2*n_skills+2, embed_size, padding_idx=0,
                                  device=self.device)
        self.rnn = nn.GRU(embed_size, hidden_dim[0]+1, n_layers[0],
                batch_first=True, device=self.device)
        self.oh_skill = nn.Linear(hidden_dim[0]+1,
                2*n_skills+1 if predict_outcome else n_skills+1,
                device=self.device)
        # self.skills = [
        #     nn.RNN(n_items+n_skills+1, hidden_dim[1], n_layers[1],
        #         batch_first=True)
        #     for _ in range(n_skills)
        # ]

    def fit(self, X, Y, T, lens):
        if self.balance:
            occs = np.zeros(
                2*self.n_skills+1 if self.predict_outcome
                else self.n_skills+1
            )
            for seq in Y:
                for val in seq:
                    if val >= 0:
                        occs[val.cpu().detach().numpy()] += 1
            most_freq = occs.max()
            occs = np.vectorize(lambda x: most_freq / x if x > 0 else 0)(occs)
            occs = torch.from_numpy(occs).float().to(self.device)
            criterion = torch.nn.CrossEntropyLoss(weight=occs, ignore_index=-1)
        else:
            criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        ls = []
        best_ls = float('inf')
        time_since_impr = 0
        cur_bsl = 0
        MAX_LEN = lens.max()
        inc_lim = 100
        print(f'bsl: {self.bsl[0]}')
        with trange(self.n_epochs, unit="epoch") as pbar:
            for epoch in pbar:
                if self.batch_size:
                    bs = (MAX_LEN + self.bsl[cur_bsl] - 1) // self.bsl[cur_bsl]

                    perm = np.random.permutation(X.shape[0])
                    batches = gen_batches(
                        X.shape[0],
                        self.batch_size
                    )
                    bl = []
                    for batch in batches:
                        x, y, t = map(lambda a: a[perm[batch]], (X, Y, T))
                        batch_size = x.size(0)
                        max_len = lens[perm[batch]].max()

                        # Clears existing gradients from previous epoch
                        optimizer.zero_grad()

                        hidden = self.init_hidden(batch_size, t)
                        for part in range((max_len + bs - 1) // bs):

                            x_embed = self.embed(x[:,part * bs:(part+1) * bs])
                            x_bs = pack_padded_sequence(
                                x_embed,
                                (lens[perm[batch]]-part*bs).clip(min=1,max=bs),
                                batch_first=True,
                                enforce_sorted=False
                            )

                            out, hidden[0] = self.rnn(
                                x_bs,
                                hidden[0]
                            )
                            y_bs = y[:,part * bs:(part+1) * bs]
                            out, _ = pad_packed_sequence(
                                out,
                                batch_first=True,
                                total_length=y_bs.shape[1]
                            )
                            output = self.oh_skill(out)

                            loss = criterion(
                                output.flatten(0, 1),
                                y_bs.flatten(0, 1)
                            )
                            hidden[0] = hidden[0].detach().clone()

                        loss.backward()
                        bl.append(loss.item())
                        optimizer.step()

                    loss = np.mean(bl)
                    if loss < best_ls:
                        time_since_impr = -1
                        best_ls = loss
                        torch.save(self.state_dict(), 'best_loss.pt')
                    ls.append(loss)
                    time_since_impr += 1
                    if time_since_impr > inc_lim:
                        cur_bsl += 1
                        if cur_bsl < len(self.bsl):
                            print(f'bsl: {self.bsl[cur_bsl]}')
                        time_since_impr = 0
                        if cur_bsl < len(self.bsl):
                            best_ls = float('inf')
                        self.load_state_dict(torch.load('best_loss.pt'))
                    if cur_bsl == len(self.bsl): # Early stopping
                        break
                else:
                    raise NotImplementedError
                    # Clears existing gradients from previous epoch
                    optimizer.zero_grad()
                    output, hidden = self(X, T)
                    loss = criterion(output.flatten(0, 1), Y.flatten(0, 1))
                    loss.backward()
                    ls.append(loss.item())
                    optimizer.step()
                    loss = loss.item()

                pbar.set_postfix(loss=f'{loss:.3f}/{best_ls:.3f}', last_inc=f'{time_since_impr}/{inc_lim}')
        print(best_ls)
        return ls

    def predict(self, X):
        return self.forward(X)[0][:,-1,0].detach().numpy()

    def forward(self, x, theta):
        batch_size = x.size(0)

        hidden = self.init_hidden(batch_size, theta)

        out, hidden[0] = self.rnn(self.embed(x), hidden[0])
        out = self.oh_skill(out)

        return out, hidden

    def gen(self, theta=0., e=None, max_n=1000):
        MAX_TOKEN = 2*self.n_skills if self.predict_outcome else self.n_skills
        while True:
            sk = []
            last = torch.tensor([2*self.n_skills]).to(self.device)
            hidden = torch.cat((
                torch.tensor([[theta] for _ in range(self.n_layers[0])])
                     .float(),
                torch.zeros(self.n_layers[0], self.hidden_dim[0])
            ), dim=1).to(self.device)
            for _ in range(max_n):
                out, hidden = self.rnn(self.embed(last), hidden)
                out = self.oh_skill(out)
                prob_dist = torch.distributions.Categorical(
                    # torch.nn.functional.softmax(out)
                    logits=out
                )
                cur = prob_dist.sample().cpu().detach().numpy()[0]
                # correct = 0
                correct_irt = 0
                skill = cur % self.n_skills if self.predict_outcome else cur
                if cur < MAX_TOKEN:
                    correct_irt = int(
                        np.random.random() < irt(theta, e[skill])
                    )
                    # if self.predict_outcome:
                    #     correct = cur // self.n_skills
                    # else:
                    #     correct = correct_irt
                if cur == MAX_TOKEN:
                    if len(sk) == 0:
                        break
                    return sk
                sk.append((skill, correct_irt, theta))
                last = torch.tensor([1 + skill + correct_irt*self.n_skills]) \
                            .to(self.device)
        assert(False)

    def init_hidden(self, batch_size, theta):
        hidden = [
            torch.cat((
                theta.reshape((1, batch_size, 1))
                     .repeat([self.n_layers[0], 1, 1]),
                torch.zeros(self.n_layers[0], batch_size, self.hidden_dim[0])
                     .to(self.device)
            ), dim=2)
        ]
        return hidden





