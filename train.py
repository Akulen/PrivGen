import matplotlib.pyplot as plt
import numpy as np
import torch

import parsing

plt.switch_backend('agg')

if __name__ == '__main__':
    parser = parsing.make_parser('Train RNN Model on a dataset')
    parsing.add_dataset_args(parser)
    parsing.add_model_params(parser)

    options = parser.parse_args()

    user = 'user'
    df, (n_user, n_skills, n_items), (theta, e) = \
        parsing.load_data(options, user)

    device = parsing.get_device(options)

    # print(df['user'].nunique(), len(theta), len(coef[0]), n_skills)
    seq = [
        (
            theta[user_id],
            [
                tuple(x)
                for x in user[['skill', 'skill_id', 'correct']].to_numpy()
            ]
        )
        for user_id, user in df.groupby(user)
    ]

    MAX_LEN = (max(len(s) for _, s in seq))
    assert(MAX_LEN < 12000-1)
    MAX_LEN += 1

    print('processing data')

    X, Y, T = [], [], []
    lens = []

    for t, s in seq:
        lens.append(len(s)+1)
        TOKEN_START = 2*n_skills
        TOKEN_IGNORE_X = 0
        TOKEN_IGNORE_Y = -1
        if options.predict_outcome:
            TOKEN_END = 2*n_skills
        else:
            TOKEN_END = n_skills
        s = [(-1, TOKEN_START, 0)] + s + [(-1, TOKEN_END, 0)]

        # X = [0] + s
        # Y = s + [MAX]

        T.append(t)
        ns = np.array([
            1 + skill + n_skills * correct
            for _, skill, correct in s[:-1]
        ])
        assert(len(ns) <= MAX_LEN)
        if len(ns) < MAX_LEN:
            ns = np.concatenate((
                ns,
                np.array([TOKEN_IGNORE_X]*(MAX_LEN-len(ns)))
            ))
        X.append(ns)
        ns = np.array([
            max(
                0,
                skill + (n_skills * correct if options.predict_outcome else 0)
            )
            for _, skill, correct in s[1:]
        ])
        if len(ns) < MAX_LEN:
            ns = np.concatenate((
                ns,
                np.array([TOKEN_IGNORE_Y]*(MAX_LEN-len(ns)))
            ))
        Y.append(ns)

    X, Y, T, lens = map(np.array, (X, Y, T, lens))
    print(X.shape, Y.shape, T.shape)

    # skills = np.array(sorted(df['skill'].unique()))
    # df_token = pd.read_csv('data/duolingo/entities.txt', names=('word',))
    # for i, line in enumerate(Y[:20]):
    #     for token in line[:lens[i]][:10]:
    #         print(df_token.loc[skills[token % n_skills], 'word']
    #                       .replace('token=', ''), end=' ')
    #     print()
    # assert(False)

    X = torch.from_numpy(X).to(device)
    Y = torch.from_numpy(Y).to(device)
    T = torch.from_numpy(T).float().to(device)

    # bsl = [20, 10, 5, 5]
    # bsl = [10, 5, 5, 2, 1]
    # bsl = [1]

    model_name, rnn = parsing.make_RNN(n_skills, n_items, device, options)

    if options.start is not None:
        rnn.load_state_dict(torch.load(options.start, map_location=device))

    ls = rnn.fit(X, Y, T, lens)

    # Keep the best training parameters for generation
    rnn.load_state_dict(torch.load('best_loss.pt', map_location=device))

    torch.save(rnn.state_dict(), f'data/{options.data}/params-{model_name}.pt')

    dpi = 96
    plt.figure(figsize=(1920/dpi, 1080/dpi), dpi=dpi)
    plt.plot(ls)
    plt.savefig(f'data/{options.data}/loss-{model_name}.png', dpi=dpi)
