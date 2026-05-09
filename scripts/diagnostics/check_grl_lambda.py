import numpy as np

N = 244400  # 100 epochs x 2444 batches/epoch

print('Epoch |   p    | alpha=1 (CURRENT)  | alpha=10 (paper g=10)')
print('------|--------|--------------------|-----------------------')
for epoch in [0, 5, 10, 17, 20, 30, 50, 75, 99]:
    iter_num = (epoch + 1) * 2444
    p = iter_num / N
    lam_1  = 2.0 / (1.0 + np.exp(-1.0  * p)) - 1.0
    lam_10 = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    print(f'  {epoch:3d}  | {p:.4f} |      {lam_1:.4f}          |       {lam_10:.4f}')

print()
print('Final lambda at end of ALL 100 epochs (p=1.0):')
lam_1_final  = 2.0 / (1.0 + np.exp(-1.0 )) - 1.0
lam_10_final = 2.0 / (1.0 + np.exp(-10.0)) - 1.0
print(f'  alpha=1  (current):  {lam_1_final:.4f}  <-- never reaches 1.0!')
print(f'  alpha=10 (paper):    {lam_10_final:.4f}  <-- correct')
