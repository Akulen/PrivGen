# from scipy.stats import norm
import numpy as np
import os
import pandas as pd
import sys

data = sys.argv[1]
gen = sys.argv[2]

df = pd.read_csv(f'data/{data}/data.csv')
coef = np.load(f'data/{data}/coef0.npy')

N_SKILLS = df['skill'].max() + 1
print(N_SKILLS)

df_gen = pd.read_csv(f'data/{data}/gen-{gen}.csv')

N_USERS = df_gen['user'].nunique()
# sampled_thetas = norm(values.mean(), values.std()).rvs(N_USERS)

df_gen['skill_diff'] = df_gen['skill'].astype(int).map(dict(zip(
    range(N_SKILLS),
    coef[0, -N_SKILLS:].reshape(-1)
)))

df_gen['item'] = 0
df_gen['wins'] = 0
df_gen['fails'] = 0

ktm_path = f'ktm/data/{data}/{gen}/'
os.makedirs(ktm_path, exist_ok=True)
df_gen.to_csv(f'{ktm_path}/data.csv', index=False)

os.chdir('ktm')
os.system(f'python encode.py --dataset {data}/{gen} --users --skills')
os.system(f'python lr.py data/{data}/{gen}/X-us.npz')
os.chdir('..')

fake_coef = np.load(f'{ktm_path}/coef0.npy')

print(N_SKILLS, df_gen['user'].nunique(), df_gen['skill'].nunique())
print(len(fake_coef[0]))
assert(len(fake_coef[0]) - (df_gen['user'].max() + 1) - N_SKILLS == 0)

fake_skill_diff = pd.DataFrame(
    range(N_SKILLS),
    columns=('skill',)
).sort_values('skill')
fake_skill_diff['skill_fake_diff'] = fake_coef[0, -N_SKILLS:]
fake_skill_diff = fake_skill_diff.set_index('skill')

# new_df = df_gen.join(fake_skill_diff, on='skill')
# print(np.sqrt(((new_df['skill_diff'] - new_df['skill_fake_diff'])**2).mean()))

original_info = df.groupby('skill').agg({'correct': ['mean', 'count']})
original_info.columns = ["_".join(a) for a in original_info.columns.to_flat_index()]
original_info['skill_diff'] = original_info.index.map(dict(zip(range(10000), coef[0, -N_SKILLS:].reshape(-1))))
original_info = original_info.merge(fake_skill_diff, on='skill')
original_info['log_count'] = original_info['correct_count'].map(np.log)
original_info['delta2'] = original_info.apply(lambda row: (row['skill_diff'] - row['skill_fake_diff']) ** 2, axis=1)
original_info['weighted_delta2'] = original_info.apply(lambda row: (row['correct_count'] * row['delta2']), axis=1)
original_info['log_weighted_delta2'] = original_info.apply(lambda row: (row['log_count'] * row['delta2']), axis=1)

FORBIDDEN = {0, 1}
selected_skills = original_info.query('correct_mean not in @FORBIDDEN').index
selected_info = original_info.query('skill in @selected_skills')
print('RMSE', selected_info['delta2'].mean() ** 0.5)
print('Weighted RMSE', (selected_info['weighted_delta2'].sum() / selected_info['correct_count'].sum()) ** 0.5)
print('Log Weighted RMSE', (selected_info['log_weighted_delta2'].sum() / selected_info['log_count'].sum()) ** 0.5)
print(f"{selected_info['delta2'].mean() ** 0.5:.3f} {(selected_info['weighted_delta2'].sum() / selected_info['correct_count'].sum()) ** 0.5:.3f} {(selected_info['log_weighted_delta2'].sum() / selected_info['log_count'].sum()) ** 0.5:.3f}", file=sys.stderr)
