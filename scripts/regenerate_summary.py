"""Regenerate summary tables and charts for newseedskcloseto16."""

import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

base = Path('outputs/newseedskcloseto16')
cutoffs = [80000, 90000, 100000, 110000, 120000, 150000, 200000]
k_values = list(range(11, 19))

# Find all seed directories
seed_dirs = {}
for d in sorted(base.iterdir()):
    if not d.is_dir() or 'seed_sweep' in d.name:
        continue
    rc = d / 'run_config.json'
    if not rc.exists():
        continue
    for p in d.name.split('_'):
        if p.startswith('s') and p[1:].isdigit():
            seed_dirs[int(p[1:])] = d
            break

seeds = sorted(seed_dirs.keys())
print(f'Using {len(seeds)} seeds: {seeds[0]}-{seeds[-1]}')


def compute_auc(metrics_path, cutoff):
    with open(metrics_path) as f:
        entries = [json.loads(l) for l in f]
    ts, rets = [], []
    for e in entries:
        if 'episode_return' in e and 'timestep' in e and e['timestep'] <= cutoff:
            ts.append(e['timestep'])
            rets.append(e['episode_return'])
    return float(np.trapezoid(rets, ts)) if len(ts) >= 2 else None


def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = wins / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / d
    return c * 100, max(0, (c - m) * 100), min(100, (c + m) * 100)


def get_avg_rank(metrics_path, t_lo=150000, t_hi=200000):
    with open(metrics_path) as f:
        entries = [json.loads(l) for l in f]
    ranks = [e['rank_mean'] for e in entries
             if 'rank_mean' in e and t_lo <= e['timestep'] <= t_hi]
    return np.mean(ranks) if ranks else None


# Compute all AUCs
print('Computing AUCs...')
results = {}
for seed, sdir in seed_dirs.items():
    ln = sdir / 'custom_dqn_noreg_ln' / 'metrics.jsonl'
    if ln.exists():
        for c in cutoffs:
            a = compute_auc(ln, c)
            if a is not None:
                results[(seed, 'ln', c)] = a
    for k in k_values:
        gb = sdir / f'custom_dqn_gradient_balanced_k{k}' / 'metrics.jsonl'
        if gb.exists():
            for c in cutoffs:
                a = compute_auc(gb, c)
                if a is not None:
                    results[(seed, f'gb_k{k}', c)] = a


# Helper to get per-cutoff stats
def get_win_count(k, cutoff):
    wins, n = 0, 0
    for s in seeds:
        if (s, 'ln', cutoff) in results and (s, f'gb_k{k}', cutoff) in results:
            n += 1
            if results[(s, f'gb_k{k}', cutoff)] > results[(s, 'ln', cutoff)]:
                wins += 1
    return wins, n


def get_gaps(k, cutoff):
    gaps = []
    for s in seeds:
        ln_key = (s, 'ln', cutoff)
        gb_key = (s, f'gb_k{k}', cutoff)
        if ln_key in results and gb_key in results and abs(results[ln_key]) > 1e-10:
            gaps.append((results[gb_key] - results[ln_key]) / abs(results[ln_key]) * 100)
    return gaps


# === 1. TABLES ===
print('Generating tables...')
cl = [f'AUC@{c // 1000}k' for c in cutoffs]
lines = [
    f'GB vs DQN+LN — WITH WARMUP (reg_warmup_frac=0.1)',
    f'Seeds: {seeds[0]}-{seeds[-1]} (n={len(seeds)})',
    '',
    'TABLE 1: Win Rate (%) of GB_k over DQN+LN  [95% Wilson CI]',
    '=' * 120,
    f'{"k":>4}  ' + '  '.join(f'{l:>14}' for l in cl),
    '-' * 120,
]
for k in k_values:
    row = f'  {k:>2}  '
    for c in cutoffs:
        wins, n = get_win_count(k, c)
        pct, lo, hi = wilson_ci(wins, n)
        row += f'  {pct:4.0f} [{lo:2.0f},{hi:2.0f}]  '
    lines.append(row)

lines += ['', '',
    'TABLE 2: Mean AUC Gap % of GB_k over DQN+LN  [±std] (95% CI)',
    '=' * 140,
    f'{"k":>4}  ' + '  '.join(f'{l:>18}' for l in cl),
    '-' * 140,
]
for k in k_values:
    row = f'  {k:>2}  '
    for c in cutoffs:
        gaps = get_gaps(k, c)
        if len(gaps) >= 2:
            m, s = np.mean(gaps), np.std(gaps, ddof=1)
            se = s / math.sqrt(len(gaps))
            row += f'  {m:+5.1f}±{s:4.1f} [{m-1.96*se:+5.1f},{m+1.96*se:+5.1f}]'
        else:
            row += f'  {"N/A":>18}'
    lines.append(row)
lines.append('')
lines.append(f'n = {len(seeds)} seeds per cell')

with open(base / 'gb_vs_ln_summary_tables.txt', 'w') as f:
    f.write('\n'.join(lines))
print(f'  -> gb_vs_ln_summary_tables.txt')


# === 2. WIN RATE CHART ===
print('Generating win rate chart...')
fig, ax = plt.subplots(figsize=(12, 6))
for k in k_values:
    wr = [wilson_ci(*get_win_count(k, c))[0] for c in cutoffs]
    ax.plot(range(len(cutoffs)), wr, 'o-', label=f'k={k}', markersize=4)
ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
ax.set_xticks(range(len(cutoffs)))
ax.set_xticklabels([f'{c//1000}k' for c in cutoffs])
ax.set_xlabel('AUC Cutoff')
ax.set_ylabel('Win Rate (%)')
ax.set_title(f'Win Rate of GB_k over DQN+LN (n={len(seeds)} seeds)')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.set_ylim(0, 100)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(base / 'winrate_by_k_across_cutoffs.png', dpi=150)
plt.close()
print(f'  -> winrate_by_k_across_cutoffs.png')


# === 3. MEAN GAP CHART ===
print('Generating mean gap chart...')
fig, ax = plt.subplots(figsize=(12, 6))
for k in k_values:
    mg, clo, chi = [], [], []
    for c in cutoffs:
        gaps = get_gaps(k, c)
        if len(gaps) >= 2:
            m, se = np.mean(gaps), np.std(gaps, ddof=1) / math.sqrt(len(gaps))
            mg.append(m); clo.append(m - 1.96 * se); chi.append(m + 1.96 * se)
        else:
            mg.append(np.nan); clo.append(np.nan); chi.append(np.nan)
    ax.plot(range(len(cutoffs)), mg, 'o-', label=f'k={k}', markersize=4)
    ax.fill_between(range(len(cutoffs)), clo, chi, alpha=0.1)
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xticks(range(len(cutoffs)))
ax.set_xticklabels([f'{c//1000}k' for c in cutoffs])
ax.set_xlabel('AUC Cutoff')
ax.set_ylabel('Mean AUC Gap % over DQN+LN')
ax.set_title(f'Mean AUC Gap of GB_k over DQN+LN (n={len(seeds)} seeds)')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(base / 'mean_gap_by_k_across_cutoffs.png', dpi=150)
plt.close()
print(f'  -> mean_gap_by_k_across_cutoffs.png')


# === 4. EFFECTIVE RANK HISTOGRAM ===
print('Generating effective rank histogram...')
ln_ranks = []
for sdir in seed_dirs.values():
    r = get_avg_rank(sdir / 'custom_dqn_noreg_ln' / 'metrics.jsonl')
    if r is not None:
        ln_ranks.append(r)
ln_ranks = np.array(ln_ranks)

gb12_ranks = []
for sdir in seed_dirs.values():
    r = get_avg_rank(sdir / 'custom_dqn_gradient_balanced_k12' / 'metrics.jsonl')
    if r is not None:
        gb12_ranks.append(r)
gb12_ranks = np.array(gb12_ranks)

print(f'  DQN+LN: n={len(ln_ranks)}, mean={np.mean(ln_ranks):.1f}, median={np.median(ln_ranks):.1f}')
print(f'  GB k=12: n={len(gb12_ranks)}, mean={np.mean(gb12_ranks):.1f}, median={np.median(gb12_ranks):.1f}')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(ln_ranks, bins=20, color='steelblue', edgecolor='white', alpha=0.8)
axes[0].axvline(np.mean(ln_ranks), color='red', ls='--', label=f'Mean={np.mean(ln_ranks):.1f}')
axes[0].axvline(np.median(ln_ranks), color='orange', ls='--', label=f'Median={np.median(ln_ranks):.1f}')
axes[0].set_xlabel('Multi-batch Effective Rank (150k-200k avg)')
axes[0].set_ylabel('Number of Seeds')
axes[0].set_title(f'DQN+LN Converged Rank (n={len(ln_ranks)})')
axes[0].legend(fontsize=8)

axes[1].hist(gb12_ranks, bins=20, color='darkorange', edgecolor='white', alpha=0.8)
axes[1].axvline(np.mean(gb12_ranks), color='red', ls='--', label=f'Mean={np.mean(gb12_ranks):.1f}')
axes[1].axvline(np.median(gb12_ranks), color='orange', ls='--', label=f'Median={np.median(gb12_ranks):.1f}')
axes[1].set_xlabel('Multi-batch Effective Rank (150k-200k avg)')
axes[1].set_ylabel('Number of Seeds')
axes[1].set_title(f'Grad-Bal k=12 Converged Rank (n={len(gb12_ranks)})')
axes[1].legend(fontsize=8)

all_vals = np.concatenate([ln_ranks, gb12_ranks])
bins = np.linspace(min(all_vals) - 0.5, max(all_vals) + 0.5, 25)
axes[2].hist(ln_ranks, bins=bins, color='steelblue', edgecolor='white', alpha=0.5,
             label=f'DQN+LN (μ={np.mean(ln_ranks):.1f})')
axes[2].hist(gb12_ranks, bins=bins, color='darkorange', edgecolor='white', alpha=0.5,
             label=f'k=12 (μ={np.mean(gb12_ranks):.1f})')
axes[2].axvline(np.mean(ln_ranks), color='steelblue', ls='--')
axes[2].axvline(np.mean(gb12_ranks), color='darkorange', ls='--')
axes[2].set_xlabel('Multi-batch Effective Rank (150k-200k avg)')
axes[2].set_ylabel('Number of Seeds')
axes[2].set_title('Comparison: DQN+LN vs k=12')
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig(base / 'effective_rank_histogram.png', dpi=150)
plt.close()
print(f'  -> effective_rank_histogram.png')

print('\nDone!')
