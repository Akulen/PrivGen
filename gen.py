from scipy.stats import norm
from tqdm import tqdm

import parsing

if __name__ == '__main__':
    parser = parsing.make_parser('Train RNN Model on a dataset')
    parsing.add_dataset_args(parser)
    parsing.add_model_params(parser)

    options = parser.parse_args()

    user = 'user'
    df, (n_user, n_skills, n_items), (theta, e) = \
        parsing.load_data(options, user)

    device = parsing.get_device(options)

    MAX_LEN = max(
        len(user[['skill', 'skill_id', 'correct']].to_numpy())
        for _, user in df.groupby(user)
    )
    assert(MAX_LEN < 12000-1)
    MAX_LEN += 1

    model_name, rnn = parsing.make_RNN(n_skills, n_items, device, options)

    rnn.eval()

    N = n_user
    S = []
    lens = []
    gen_theta = norm(theta.mean(), theta.std()).rvs(N)

    skills = sorted(df['skill'].unique())

    for i in tqdm(range(N)):
        s = rnn.gen(theta=gen_theta[i], e=e[skills], max_n=MAX_LEN)
        S.append(s)
        # print(len(s), s[:10])
        lens.append(len(s))

    print(sum(lens))

    with open(f'data/{options.data}/gen-{model_name}.csv', 'w') as f:
        print('user,skill_id,skill,correct,theta', file=f)
        for i, s in enumerate(S):
            for skill_id, correct, theta in s:
                skill = skills[skill_id]
                print(f'{i},{skill_id},{skill},{correct},{theta:.6}', file=f)
