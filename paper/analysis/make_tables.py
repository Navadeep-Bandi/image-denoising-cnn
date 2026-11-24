"""Generate LaTeX table fragments + paired statistics from the archived per-case CSVs.
Usage (from paper/analysis): python make_tables.py [results_dir=../../results] [out_dir=../tables]
"""
import sys, pandas as pd, numpy as np
from scipy import stats
R = (sys.argv[1] if len(sys.argv) > 1 else '../../results').rstrip('/\\')
OUT = (sys.argv[2] if len(sys.argv) > 2 else '../tables').rstrip('/\\')
E = R + '/experiments/'
rng = np.random.default_rng(0)
K = ['image', 'noise_type', 'severity']
NN = {'gaussian': 'Gaussian', 'sp': 'Salt-pepper', 'speckle': 'Speckle'}
def boot(d, n=5000):
    return np.percentile([rng.choice(d, len(d)).mean() for _ in range(n)], [2.5, 97.5])
def pv(p): return '$<10^{-6}$' if p < 1e-6 else f'{p:.3f}'
def sg(x, nd=3): return f'{x:+.{nd}f}'
def write(name, s): open(f'{OUT}/{name}', 'w').write(s + '\\bottomrule\n')  # rule inside fragment: avoids a LaTeX \input/\bottomrule clash

# ---------- Table: paired statistics (Exp1 blending + Exp3 overlap sweep)
e1 = pd.read_csv(E + 'exp1/exp1_blending_comparison.csv')
P = e1.pivot_table(index=K, columns='config', values=['psnr', 'ssim'])
e3 = pd.read_csv(E + 'exp3/exp3_overlap_sweep.csv')
Q = e3.pivot_table(index=K, columns='overlap_px', values=['psnr', 'ssim'])
rows = [('Uniform 25\\% vs.\\ no overlap', P['psnr']['B_uniform_25pct'], P['psnr']['A_no_overlap'], P['ssim']['B_uniform_25pct'], P['ssim']['A_no_overlap']),
        ('Hann 25\\% vs.\\ uniform 25\\%', P['psnr']['C_hann_25pct'], P['psnr']['B_uniform_25pct'], P['ssim']['C_hann_25pct'], P['ssim']['B_uniform_25pct']),
        ('Hann 25\\% vs.\\ no overlap', P['psnr']['C_hann_25pct'], P['psnr']['A_no_overlap'], P['ssim']['C_hann_25pct'], P['ssim']['A_no_overlap']),
        ('12.5\\% vs.\\ 0\\% overlap', Q['psnr'][16], Q['psnr'][0], Q['ssim'][16], Q['ssim'][0]),
        ('25\\% vs.\\ 12.5\\% overlap', Q['psnr'][32], Q['psnr'][16], Q['ssim'][32], Q['ssim'][16]),
        ('37.5\\% vs.\\ 25\\% overlap', Q['psnr'][48], Q['psnr'][32], Q['ssim'][48], Q['ssim'][32]),
        ('50\\% vs.\\ 25\\% overlap', Q['psnr'][64], Q['psnr'][32], Q['ssim'][64], Q['ssim'][32])]
s = ''
for name, a, b, sa, sb in rows:
    d = (a - b).values; lo, hi = boot(d)
    s += f"{name} & {sg(d.mean(),4)} & [{lo:+.4f}, {hi:+.4f}] & {np.mean(d>0)*100:.0f}\\% & {pv(stats.wilcoxon(d).pvalue)} & {sg((sa-sb).mean(),4)}\\\\\n"
write('tab_stats_rows.tex', s)

# ---------- Table: per noise x severity (pipeline), Basic CNN gain, Hann gain
e5 = pd.read_csv(E + 'exp5/exp5_restoration_results.csv')
e4 = pd.read_csv(E + 'exp4/exp4_efficiency.csv')
b = e4[e4.config == '1_basic_cnn'].set_index(K).psnr
e5i = e5.set_index(K); bgain = (b - e5i.noisy_psnr)
hg = (P['psnr']['C_hann_25pct'] - P['psnr']['A_no_overlap'])
s = ''
for n in ['gaussian', 'sp', 'speckle']:
    for sev in ['low', 'medium', 'high']:
        m = e5[(e5.noise_type == n) & (e5.severity == sev)]
        bg = bgain.xs((n, sev), level=('noise_type', 'severity')).mean()
        h = hg.xs((n, sev), level=('noise_type', 'severity')).mean()
        s += f"{NN[n] if sev=='low' else ''} & {sev.capitalize()} & {m.noisy_psnr.mean():.2f} & {m.restored_psnr.mean():.2f} & {m.psnr_improvement.mean():+.2f} & {m.restored_ssim.mean():.3f} & {bg:+.2f} & {h:+.3f}\\\\\n"
    if n != 'speckle': s += '\\midrule\n'
write('tab_severity_rows.tex', s)
print('basic negative-gain cases', int((bgain < 0).sum()), '/', len(bgain))

# ---------- Table: worst cases
w = e5.nsmallest(5, 'restored_psnr')
s = ''.join(f"{r.image[:4]} & {NN[r.noise_type]} & {r.severity.capitalize()} & {r.noisy_psnr:.2f} & {r.restored_psnr:.2f} & {r.restored_ssim:.3f}\\\\\n" for r in w.itertuples())
write('tab_worst_rows.tex', s)
im = e5.groupby('image').psnr_improvement.mean().sort_values()
print('lowest mean-gain images:', im.head(3).round(2).to_dict(), 'mean', im.mean().round(2))

# ---------- Table: 100-image, 128x128 comparison (all_models_summary)
a = pd.read_csv(R + '/all_models_summary.csv')
s = ''
for n in ['Gaussian', 'Salt-and-pepper', 'Speckle']:
    for sev in ['Low', 'Medium', 'High']:
        x = a[(a.noise_type == n) & (a.severity == sev)].set_index('model')
        s += f"{n.replace('Salt-and-pepper','Salt-pepper') if sev=='Low' else ''} & {sev} & {x.loc['Basic CNN','noisy_psnr']:.2f} & {x.loc['Basic CNN','denoised_psnr']:.2f} & {x.loc['DnCNN','denoised_psnr']:.2f} & {x.loc['Basic CNN','denoised_ssim']:.3f} & {x.loc['DnCNN','denoised_ssim']:.3f}\\\\\n"
    if n != 'Speckle': s += '\\midrule\n'
write('tab_allmodels_rows.tex', s)

# ---------- misc numbers
d = (e4.pivot_table(index=K, columns='config', values='mean_sec'))
dd = (d['5_auto_pipeline'] - d['4_dncnn_hann_25pct']).values; lo, hi = boot(dd)
print('auto-hann time', dd.mean().round(4), lo.round(4), hi.round(4), 'pct', (dd.mean()/d['4_dncnn_hann_25pct'].mean()*100).round(1))
e2 = pd.read_csv(E + 'exp2/exp2_boundary_interior.csv')
for n, g in e2.groupby('noise_type'):
    x = g.delta_psnr_interior_minus_boundary.values; lo, hi = boot(x)
    print(n, round(x.mean(), 4), [round(lo, 4), round(hi, 4)], f'{(x>0).mean()*100:.0f}%')
x = e2.delta_psnr_interior_minus_boundary.values; print('all', x.mean().round(4), boot(x).round(4), (x > 0).mean())
