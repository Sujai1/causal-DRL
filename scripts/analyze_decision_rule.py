"""Analyze adaptive decision rules for selecting GB rank bound k.

For each seed, uses the DQN+LN effective rank trajectory to pick which
GB_k result to use, then compares this 'adaptive' baseline against DQN+LN.

Decision rules:
  - Adaptive-25k: k = ceil(mean effective rank of LN over last 25k steps), clamped to [11,18]
  - Adaptive-50k: k = ceil(mean effective rank of LN over last 50k steps), clamped to [11,18]
"""

import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

base = Path('outputs/newseedskcloseto16')
cutoffs = [80000, 90000, 100000, 110000, 120000, 150000, 200000]
k_values = list(range(11, 19))
K_MIN, K_MAX = 11, 18
ROLLING_WINDOW = 50
FINAL_WINDOW = 100
ADAPTIVE_WINDOWS = [25000, 50000]  # timestep windows for averaging effective rank

# ── Find all seed directories ──
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


# ── Load episode data ──
def load_episodes(metrics_path):
    """Return list of (timestep, episode_return) tuples."""
    with open(metrics_path) as f:
        entries = [json.loads(l) for l in f]
    return [(e['timestep'], e['episode_return'])
            for e in entries if 'episode_return' in e and 'timestep' in e]


def load_effective_ranks(metrics_path):
    """Return list of (timestep, effective_rank) tuples."""
    with open(metrics_path) as f:
        entries = [json.loads(l) for l in f]
    return [(e['timestep'], e['effective_rank'])
            for e in entries if 'effective_rank' in e and 'timestep' in e]


print('Loading data...')
episode_data = {}   # (seed, method) -> [(timestep, return), ...]
ln_eff_ranks = {}   # seed -> [(timestep, effective_rank), ...]

for seed, sdir in seed_dirs.items():
    ln_path = sdir / 'custom_dqn_noreg_ln' / 'metrics.jsonl'
    if ln_path.exists():
        episode_data[(seed, 'ln')] = load_episodes(ln_path)
        ln_eff_ranks[seed] = load_effective_ranks(ln_path)
    for k in k_values:
        gb_path = sdir / f'custom_dqn_gradient_balanced_k{k}' / 'metrics.jsonl'
        if gb_path.exists():
            episode_data[(seed, f'gb_k{k}')] = load_episodes(gb_path)


# ── Compute adaptive k selection for each seed ──
def select_k_adaptive(eff_rank_trajectory, window_timesteps):
    """Select k = ceil(mean effective rank over last `window_timesteps` steps), clamped."""
    max_ts = max(t for t, _ in eff_rank_trajectory)
    threshold = max_ts - window_timesteps
    recent = [r for t, r in eff_rank_trajectory if t >= threshold]
    if not recent:
        return None
    mean_rank = np.mean(recent)
    k = int(math.ceil(mean_rank))
    return max(K_MIN, min(K_MAX, k))


# Build adaptive selections
adaptive_selections = {}  # (seed, window) -> selected_k
for window in ADAPTIVE_WINDOWS:
    label = f'{window // 1000}k'
    k_counts = Counter()
    for seed in seeds:
        ranks = ln_eff_ranks.get(seed)
        if ranks is None:
            continue
        k = select_k_adaptive(ranks, window)
        if k is not None:
            adaptive_selections[(seed, window)] = k
            k_counts[k] += 1

    print(f'\nAdaptive-{label} k distribution:')
    for k in sorted(k_counts.keys()):
        print(f'  k={k}: {k_counts[k]} seeds ({k_counts[k]/len(seeds)*100:.1f}%)')

# Also build episode data for the adaptive baselines
# For each seed, the adaptive baseline uses the GB_k result where k was selected
for window in ADAPTIVE_WINDOWS:
    for seed in seeds:
        k = adaptive_selections.get((seed, window))
        if k is not None:
            source = episode_data.get((seed, f'gb_k{k}'))
            if source is not None:
                episode_data[(seed, f'adaptive_{window // 1000}k')] = source


# ── Metric computation functions ──

def compute_auc(episodes, cutoff):
    ts = [t for t, r in episodes if t <= cutoff]
    rets = [r for t, r in episodes if t <= cutoff]
    return float(np.trapezoid(rets, ts)) if len(ts) >= 2 else None


def compute_final_performance(episodes, cutoff, window=FINAL_WINDOW):
    filtered = [r for t, r in episodes if t <= cutoff]
    if len(filtered) < window:
        return None
    return float(np.mean(filtered[-window:]))


def compute_best_rolling(episodes, cutoff, window=ROLLING_WINDOW):
    filtered = [r for t, r in episodes if t <= cutoff]
    if len(filtered) < window:
        return None
    rolling = np.convolve(filtered, np.ones(window)/window, mode='valid')
    return float(np.max(rolling))


def compute_time_to_threshold(episodes, threshold, max_timestep=200000, window=ROLLING_WINDOW):
    rets = [r for _, r in episodes]
    ts = [t for t, _ in episodes]
    if len(rets) < window:
        return max_timestep
    rolling = np.convolve(rets, np.ones(window)/window, mode='valid')
    for i, val in enumerate(rolling):
        if val >= threshold:
            return ts[i + window - 1]
    return max_timestep


# ── Stat helpers ──

def wilson_ci(wins, n, z=1.96):
    if n == 0:
        return 0, 0, 0
    p = wins / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    m = z * math.sqrt((p * (1 - p) + z**2 / (4 * n)) / n) / d
    return c * 100, max(0, (c - m) * 100), min(100, (c + m) * 100)


# ── Comparison logic ──

# All methods to compare: fixed k values + adaptive baselines
all_methods = [f'gb_k{k}' for k in k_values] + [f'adaptive_{w // 1000}k' for w in ADAPTIVE_WINDOWS]
method_labels = {f'gb_k{k}': f'GB k={k}' for k in k_values}
for w in ADAPTIVE_WINDOWS:
    method_labels[f'adaptive_{w // 1000}k'] = f'Adaptive-{w // 1000}k'


def get_win_and_gaps_method(results, method, cutoff, higher_is_better=True):
    """Get wins count, n, and gap percentages for a given method vs LN."""
    wins, n, gaps = 0, 0, []
    for s in seeds:
        ln_val = results.get((s, 'ln', cutoff))
        m_val = results.get((s, method, cutoff))
        if ln_val is not None and m_val is not None:
            n += 1
            if higher_is_better:
                if m_val > ln_val:
                    wins += 1
            else:
                if m_val < ln_val:
                    wins += 1
            if abs(ln_val) > 1e-10:
                gap_pct = (m_val - ln_val) / abs(ln_val) * 100
                if not higher_is_better:
                    gap_pct = -gap_pct
                gaps.append(gap_pct)
    return wins, n, gaps


def compute_all_results(metric_fn):
    """Compute metric for all seeds and all methods."""
    results = {}
    for seed in seeds:
        for method in ['ln'] + all_methods:
            eps = episode_data.get((seed, method))
            if eps is None:
                continue
            for cutoff in cutoffs:
                val = metric_fn(eps, cutoff)
                if val is not None:
                    results[(seed, method, cutoff)] = val
    return results


def generate_table_and_charts(results, metric_name, higher_is_better=True):
    """Generate tables and charts including adaptive baselines."""
    cl = [f'{c // 1000}k' for c in cutoffs]
    methods_to_show = [f'gb_k{k}' for k in k_values] + [f'adaptive_{w // 1000}k' for w in ADAPTIVE_WINDOWS]

    # ── Table ──
    lines = [
        f'{metric_name} — GB vs DQN+LN (n={len(seeds)} seeds)',
        f'Includes adaptive baselines that select k per-seed based on LN effective rank',
        '',
        f'Win Rate (%) [95% Wilson CI]    (GB {">" if higher_is_better else "<"} LN = win)',
        '=' * 120,
        f'{"Method":>16}  ' + '  '.join(f'{l:>14}' for l in cl),
        '-' * 120,
    ]
    for method in methods_to_show:
        label = method_labels[method]
        row = f'{label:>16}  '
        for c in cutoffs:
            wins, n, _ = get_win_and_gaps_method(results, method, c, higher_is_better)
            pct, lo, hi = wilson_ci(wins, n)
            row += f'  {pct:4.0f} [{lo:2.0f},{hi:2.0f}]  '
        lines.append(row)

    lines += ['', '',
        f'Mean Gap % [±std] (95% CI)    (positive = GB better)',
        '=' * 150,
        f'{"Method":>16}  ' + '  '.join(f'{l:>18}' for l in cl),
        '-' * 150,
    ]
    for method in methods_to_show:
        label = method_labels[method]
        row = f'{label:>16}  '
        for c in cutoffs:
            _, _, gaps = get_win_and_gaps_method(results, method, c, higher_is_better)
            if len(gaps) >= 2:
                m, s = np.mean(gaps), np.std(gaps, ddof=1)
                se = s / math.sqrt(len(gaps))
                row += f'  {m:+5.1f}±{s:4.1f} [{m-1.96*se:+5.1f},{m+1.96*se:+5.1f}]'
            else:
                row += f'  {"N/A":>18}'
        lines.append(row)
    lines.append('')

    safe_name = 'adaptive_' + metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')

    with open(base / f'{safe_name}_tables.txt', 'w') as f:
        f.write('\n'.join(lines))
    print(f'  -> {safe_name}_tables.txt')

    # ── Win Rate Chart ──
    fig, ax = plt.subplots(figsize=(13, 7))
    # Fixed k: thin gray lines
    for k in k_values:
        method = f'gb_k{k}'
        wr = []
        for c in cutoffs:
            wins, n, _ = get_win_and_gaps_method(results, method, c, higher_is_better)
            wr.append(wilson_ci(wins, n)[0])
        ax.plot(range(len(cutoffs)), wr, '-', color='gray', alpha=0.3,
                linewidth=1, label=f'k={k}' if k == k_values[0] else None)
        # Annotate last point
        ax.annotate(f'k={k}', (len(cutoffs)-1, wr[-1]), fontsize=6,
                    color='gray', alpha=0.5, ha='left', va='center',
                    xytext=(5, 0), textcoords='offset points')

    # Adaptive: bold colored lines
    colors = ['#e41a1c', '#377eb8']
    for i, w in enumerate(ADAPTIVE_WINDOWS):
        method = f'adaptive_{w // 1000}k'
        wr = []
        for c in cutoffs:
            wins, n, _ = get_win_and_gaps_method(results, method, c, higher_is_better)
            wr.append(wilson_ci(wins, n)[0])
        ax.plot(range(len(cutoffs)), wr, 'o-', color=colors[i],
                linewidth=2.5, markersize=6, label=f'Adaptive-{w//1000}k')

    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(cutoffs)))
    ax.set_xticklabels(cl)
    ax.set_xlabel('Timestep Cutoff')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(f'{metric_name}: Adaptive vs DQN+LN (n={len(seeds)})\n'
                 f'Gray = fixed k, colored = adaptive')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(30, 85)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(base / f'{safe_name}_winrate.png', dpi=150)
    plt.close()
    print(f'  -> {safe_name}_winrate.png')

    # ── Mean Gap Chart ──
    fig, ax = plt.subplots(figsize=(13, 7))
    for k in k_values:
        method = f'gb_k{k}'
        mg = []
        for c in cutoffs:
            _, _, gaps = get_win_and_gaps_method(results, method, c, higher_is_better)
            mg.append(np.mean(gaps) if gaps else np.nan)
        ax.plot(range(len(cutoffs)), mg, '-', color='gray', alpha=0.3, linewidth=1)
        ax.annotate(f'k={k}', (len(cutoffs)-1, mg[-1]), fontsize=6,
                    color='gray', alpha=0.5, ha='left', va='center',
                    xytext=(5, 0), textcoords='offset points')

    for i, w in enumerate(ADAPTIVE_WINDOWS):
        method = f'adaptive_{w // 1000}k'
        mg, clo, chi = [], [], []
        for c in cutoffs:
            _, _, gaps = get_win_and_gaps_method(results, method, c, higher_is_better)
            if len(gaps) >= 2:
                m, se = np.mean(gaps), np.std(gaps, ddof=1) / math.sqrt(len(gaps))
                mg.append(m); clo.append(m - 1.96 * se); chi.append(m + 1.96 * se)
            else:
                mg.append(np.nan); clo.append(np.nan); chi.append(np.nan)
        ax.plot(range(len(cutoffs)), mg, 'o-', color=colors[i],
                linewidth=2.5, markersize=6, label=f'Adaptive-{w//1000}k')
        ax.fill_between(range(len(cutoffs)), clo, chi, alpha=0.15, color=colors[i])

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(cutoffs)))
    ax.set_xticklabels(cl)
    ax.set_xlabel('Timestep Cutoff')
    ax.set_ylabel('Mean Gap % (positive = GB better)')
    ax.set_title(f'{metric_name}: Mean Gap — Adaptive vs DQN+LN (n={len(seeds)})\n'
                 f'Gray = fixed k, colored = adaptive')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(base / f'{safe_name}_meangap.png', dpi=150)
    plt.close()
    print(f'  -> {safe_name}_meangap.png')


# ══════════════════════════════════════════════════
# Generate all metrics
# ══════════════════════════════════════════════════

print('\n1. AUC of Episode Returns')
auc_results = compute_all_results(compute_auc)
generate_table_and_charts(auc_results, 'AUC of Episode Returns', higher_is_better=True)

print(f'\n2. Final Performance (last {FINAL_WINDOW} episodes)')
final_results = compute_all_results(
    lambda eps, c: compute_final_performance(eps, c, FINAL_WINDOW))
generate_table_and_charts(final_results, f'Final Performance (last {FINAL_WINDOW} eps)', higher_is_better=True)

print(f'\n3. Best Rolling Mean (window={ROLLING_WINDOW})')
best_results = compute_all_results(
    lambda eps, c: compute_best_rolling(eps, c, ROLLING_WINDOW))
generate_table_and_charts(best_results, f'Best Rolling Mean (w={ROLLING_WINDOW})', higher_is_better=True)

# ── K selection distribution chart ──
print('\n4. K selection distribution')
fig, axes = plt.subplots(1, len(ADAPTIVE_WINDOWS), figsize=(12, 5))
for i, w in enumerate(ADAPTIVE_WINDOWS):
    ax = axes[i]
    selected_ks = [adaptive_selections[(s, w)] for s in seeds if (s, w) in adaptive_selections]
    counts = Counter(selected_ks)
    x = list(range(K_MIN, K_MAX + 1))
    heights = [counts.get(k, 0) for k in x]
    ax.bar(x, heights, color='steelblue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Selected k')
    ax.set_ylabel('Number of seeds')
    ax.set_title(f'Adaptive-{w//1000}k: k distribution (n={len(selected_ks)})')
    ax.set_xticks(x)
    # Add mean line
    mean_k = np.mean(selected_ks)
    ax.axvline(mean_k, color='red', linestyle='--', linewidth=1.5, label=f'mean={mean_k:.1f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(base / 'adaptive_k_distribution.png', dpi=150)
plt.close()
print(f'  -> adaptive_k_distribution.png')

# ── Per-seed details: what k was selected and the mean effective rank ──
print('\n5. Per-seed adaptive selection details')
detail_lines = ['Adaptive k selection details per seed', '']
for w in ADAPTIVE_WINDOWS:
    label = f'{w // 1000}k'
    detail_lines.append(f'--- Adaptive-{label} ---')
    detail_lines.append(f'{"Seed":>6}  {"Mean Eff Rank":>14}  {"Selected k":>10}')
    detail_lines.append('-' * 36)
    for seed in seeds:
        ranks = ln_eff_ranks.get(seed)
        if ranks is None:
            continue
        max_ts = max(t for t, _ in ranks)
        threshold = max_ts - w
        recent = [r for t, r in ranks if t >= threshold]
        mean_rank = np.mean(recent) if recent else float('nan')
        k = adaptive_selections.get((seed, w), '?')
        detail_lines.append(f'{seed:>6}  {mean_rank:>14.2f}  {k:>10}')
    detail_lines.append('')

with open(base / 'adaptive_selection_details.txt', 'w') as f:
    f.write('\n'.join(detail_lines))
print(f'  -> adaptive_selection_details.txt')

print('\nAll done!')
