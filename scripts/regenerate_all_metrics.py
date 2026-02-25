"""Regenerate summary tables and charts for multiple RL metrics.

Includes adaptive baselines that select GB rank bound k per-seed based on
the DQN+LN effective rank trajectory.
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
ROLLING_WINDOW = 50  # episodes for rolling mean
FINAL_WINDOW = 100   # last N episodes for "final performance"
ADAPTIVE_WINDOWS = [25000, 50000]  # timestep windows for LN effective rank averaging (from end)
ADAPTIVE_EARLY = [(30000, 40000), (35000, 40000)]  # (start, end) timestep ranges for early adaptive

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


# ── Load all episode data ──
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


print('Loading episode data...')
episode_data = {}   # (seed, method) -> [(timestep, return), ...]
ln_eff_ranks = {}   # seed -> [(timestep, effective_rank), ...]
for seed, sdir in seed_dirs.items():
    ln = sdir / 'custom_dqn_noreg_ln' / 'metrics.jsonl'
    if ln.exists():
        episode_data[(seed, 'ln')] = load_episodes(ln)
        ln_eff_ranks[seed] = load_effective_ranks(ln)
    for k in k_values:
        gb = sdir / f'custom_dqn_gradient_balanced_k{k}' / 'metrics.jsonl'
        if gb.exists():
            episode_data[(seed, f'gb_k{k}')] = load_episodes(gb)


# ── Adaptive k selection ──
def select_k_adaptive(eff_rank_trajectory, window_timesteps):
    """Select k = ceil(mean effective rank over last `window_timesteps`), clamped to [K_MIN, K_MAX]."""
    max_ts = max(t for t, _ in eff_rank_trajectory)
    threshold = max_ts - window_timesteps
    recent = [r for t, r in eff_rank_trajectory if t >= threshold]
    if not recent:
        return None
    return max(K_MIN, min(K_MAX, int(math.ceil(np.mean(recent)))))


adaptive_selections = {}  # (seed, window) -> selected_k
adaptive_labels = {}      # method_key -> display label
for window in ADAPTIVE_WINDOWS:
    label = f'{window // 1000}k'
    adaptive_labels[f'adaptive_{label}'] = f'Adapt-{label}'
    k_counts = Counter()
    for seed in seeds:
        ranks = ln_eff_ranks.get(seed)
        if ranks is None:
            continue
        k = select_k_adaptive(ranks, window)
        if k is not None:
            adaptive_selections[(seed, window)] = k
            k_counts[k] += 1
            # Point to the corresponding GB result
            source = episode_data.get((seed, f'gb_k{k}'))
            if source is not None:
                episode_data[(seed, f'adaptive_{label}')] = source

    print(f'  Adaptive-{label} k distribution: ' +
          ', '.join(f'k={k}:{k_counts[k]}' for k in sorted(k_counts)))


# ── Early adaptive k selection (fixed timestep windows) ──
def select_k_early(eff_rank_trajectory, ts_start, ts_end):
    """Select k = ceil(mean effective rank in [ts_start, ts_end)), clamped to [K_MIN, K_MAX]."""
    vals = [r for t, r in eff_rank_trajectory if ts_start <= t < ts_end]
    if not vals:
        return None
    return max(K_MIN, min(K_MAX, int(math.ceil(np.mean(vals)))))


for ts_start, ts_end in ADAPTIVE_EARLY:
    label = f'early_{ts_start // 1000}k-{ts_end // 1000}k'
    display = f'Early{ts_start // 1000}-{ts_end // 1000}k'
    adaptive_labels[label] = display
    k_counts = Counter()
    for seed in seeds:
        ranks = ln_eff_ranks.get(seed)
        if ranks is None:
            continue
        k = select_k_early(ranks, ts_start, ts_end)
        if k is not None:
            adaptive_selections[(seed, (ts_start, ts_end))] = k
            k_counts[k] += 1
            source = episode_data.get((seed, f'gb_k{k}'))
            if source is not None:
                episode_data[(seed, label)] = source

    print(f'  {display} k distribution: ' +
          ', '.join(f'k={k}:{k_counts[k]}' for k in sorted(k_counts)))

early_method_keys = [f'early_{s // 1000}k-{e // 1000}k' for s, e in ADAPTIVE_EARLY]
adaptive_method_keys = [f'adaptive_{w // 1000}k' for w in ADAPTIVE_WINDOWS] + early_method_keys

# Oracle will be added per-metric (best k depends on the metric/cutoff)
ORACLE_KEY = 'oracle'
adaptive_labels[ORACLE_KEY] = 'Oracle'


# ── Metric computation functions ──

def compute_auc(episodes, cutoff):
    """Trapezoidal AUC of episode returns up to cutoff."""
    ts = [t for t, r in episodes if t <= cutoff]
    rets = [r for t, r in episodes if t <= cutoff]
    return float(np.trapezoid(rets, ts)) if len(ts) >= 2 else None


def compute_final_performance(episodes, cutoff, window=FINAL_WINDOW):
    """Mean return of last `window` episodes before cutoff."""
    filtered = [r for t, r in episodes if t <= cutoff]
    if len(filtered) < window:
        return None
    return float(np.mean(filtered[-window:]))


def compute_best_rolling(episodes, cutoff, window=ROLLING_WINDOW):
    """Best rolling-mean return achieved up to cutoff."""
    filtered = [r for t, r in episodes if t <= cutoff]
    if len(filtered) < window:
        return None
    rolling = np.convolve(filtered, np.ones(window)/window, mode='valid')
    return float(np.max(rolling))


def compute_time_to_threshold(episodes, threshold, max_timestep=200000, window=ROLLING_WINDOW):
    """First timestep where rolling mean exceeds threshold. Returns max_timestep if never."""
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


def compute_comparison(metric_fn, higher_is_better=True):
    """Compute metric for all seeds/methods, including oracle.
    Returns dict: (seed, method, cutoff) -> value."""
    all_method_keys = ['ln'] + [f'gb_k{k}' for k in k_values] + adaptive_method_keys
    results = {}
    for seed in seeds:
        for method_key in all_method_keys:
            eps = episode_data.get((seed, method_key))
            if eps is None:
                continue
            for cutoff in cutoffs:
                val = metric_fn(eps, cutoff)
                if val is not None:
                    results[(seed, method_key, cutoff)] = val

    # Oracle: for each (seed, cutoff), pick the GB_k with the best metric value
    for seed in seeds:
        for cutoff in cutoffs:
            best_val, best_k = None, None
            for k in k_values:
                val = results.get((seed, f'gb_k{k}', cutoff))
                if val is None:
                    continue
                if best_val is None or (higher_is_better and val > best_val) or \
                   (not higher_is_better and val < best_val):
                    best_val, best_k = val, k
            if best_val is not None:
                results[(seed, ORACLE_KEY, cutoff)] = best_val
    return results


def compute_ttt_comparison(threshold):
    """Time-to-threshold comparison (lower is better)."""
    results = {}
    for seed in seeds:
        for method_key in ['ln'] + [f'gb_k{k}' for k in k_values]:
            eps = episode_data.get((seed, method_key))
            if eps is None:
                continue
            val = compute_time_to_threshold(eps, threshold)
            results[(seed, method_key)] = val
    return results


def get_win_and_gaps(results, method_key, cutoff, higher_is_better=True):
    """Get wins count, n, and gap percentages for method_key vs LN."""
    wins, n, gaps = 0, 0, []
    for s in seeds:
        ln_val = results.get((s, 'ln', cutoff))
        m_val = results.get((s, method_key, cutoff))
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
                    gap_pct = -gap_pct  # flip so positive = GB better
                gaps.append(gap_pct)
    return wins, n, gaps


def generate_table_and_charts(results, metric_name, higher_is_better=True):
    """Generate summary table, win rate chart, and mean gap chart for a metric.
    Includes both fixed-k GB methods and adaptive baselines."""
    cl = [f'{c // 1000}k' for c in cutoffs]

    # Build ordered list of (method_key, display_label) pairs
    method_rows = [(f'gb_k{k}', f'k={k:>2}') for k in k_values]
    for mk in adaptive_method_keys:
        method_rows.append((mk, adaptive_labels[mk]))
    method_rows.append((ORACLE_KEY, adaptive_labels[ORACLE_KEY]))

    # ── Table ──
    lines = [
        f'{metric_name} — GB vs DQN+LN (n={len(seeds)} seeds)',
        '',
        f'Win Rate (%) [95% Wilson CI]    (GB {">" if higher_is_better else "<"} LN = win)',
        '=' * 130,
        f'{"Method":>12}  ' + '  '.join(f'{l:>14}' for l in cl),
        '-' * 130,
    ]
    for method_key, label in method_rows:
        row = f'{label:>12}  '
        for c in cutoffs:
            wins, n, _ = get_win_and_gaps(results, method_key, c, higher_is_better)
            pct, lo, hi = wilson_ci(wins, n)
            row += f'  {pct:4.0f} [{lo:2.0f},{hi:2.0f}]  '
        # Separator before adaptive rows
        if method_key == adaptive_method_keys[0] and lines[-1] != '':
            lines.append('')
        lines.append(row)

    lines += ['', '',
        f'Mean Gap % [±std] (95% CI)    (positive = GB better)',
        '=' * 150,
        f'{"Method":>12}  ' + '  '.join(f'{l:>18}' for l in cl),
        '-' * 150,
    ]
    for method_key, label in method_rows:
        row = f'{label:>12}  '
        for c in cutoffs:
            _, _, gaps = get_win_and_gaps(results, method_key, c, higher_is_better)
            if len(gaps) >= 2:
                m, s = np.mean(gaps), np.std(gaps, ddof=1)
                se = s / math.sqrt(len(gaps))
                row += f'  {m:+5.1f}±{s:4.1f} [{m-1.96*se:+5.1f},{m+1.96*se:+5.1f}]'
            else:
                row += f'  {"N/A":>18}'
        if method_key == adaptive_method_keys[0] and lines[-1] != '':
            lines.append('')
        lines.append(row)
    lines.append('')

    safe_name = metric_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')

    with open(base / f'{safe_name}_tables.txt', 'w') as f:
        f.write('\n'.join(lines))
    print(f'  -> {safe_name}_tables.txt')

    # ── Win Rate Chart ──
    fig, ax = plt.subplots(figsize=(13, 7))
    # Fixed k: thin gray lines
    for k in k_values:
        mk = f'gb_k{k}'
        wr = []
        for c in cutoffs:
            wins, n, _ = get_win_and_gaps(results, mk, c, higher_is_better)
            wr.append(wilson_ci(wins, n)[0])
        ax.plot(range(len(cutoffs)), wr, '-', color='gray', alpha=0.35, linewidth=1,
                label=f'Fixed k (k={k_values[0]}-{k_values[-1]})' if k == k_values[0] else None)
        ax.annotate(f'k={k}', (len(cutoffs)-1, wr[-1]), fontsize=6,
                    color='gray', alpha=0.6, ha='left', va='center',
                    xytext=(5, 0), textcoords='offset points')

    # Adaptive: bold colored lines
    adapt_colors = {'adaptive_25k': '#e41a1c', 'adaptive_50k': '#377eb8',
                     'early_30k-40k': '#4daf4a', 'early_35k-40k': '#984ea3'}
    for mk in adaptive_method_keys:
        wr = []
        for c in cutoffs:
            wins, n, _ = get_win_and_gaps(results, mk, c, higher_is_better)
            wr.append(wilson_ci(wins, n)[0])
        ax.plot(range(len(cutoffs)), wr, 'o-', color=adapt_colors[mk],
                linewidth=2.5, markersize=6, label=adaptive_labels[mk])

    # Oracle: dashed gold line
    oracle_wr = []
    for c in cutoffs:
        wins, n, _ = get_win_and_gaps(results, ORACLE_KEY, c, higher_is_better)
        oracle_wr.append(wilson_ci(wins, n)[0])
    ax.plot(range(len(cutoffs)), oracle_wr, 's--', color='#ff7f00',
            linewidth=2, markersize=6, label='Oracle (best k)')

    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(cutoffs)))
    ax.set_xticklabels(cl)
    ax.set_xlabel('Timestep Cutoff')
    ax.set_ylabel('Win Rate (%)')
    ax.set_title(f'{metric_name}: Win Rate of GB over DQN+LN (n={len(seeds)})')
    ax.legend(loc='upper left', fontsize=9)
    ax.set_ylim(30, 100)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(base / f'{safe_name}_winrate.png', dpi=150)
    plt.close()
    print(f'  -> {safe_name}_winrate.png')

    # ── Mean Gap Chart ──
    fig, ax = plt.subplots(figsize=(13, 7))
    for k in k_values:
        mk = f'gb_k{k}'
        mg = []
        for c in cutoffs:
            _, _, gaps = get_win_and_gaps(results, mk, c, higher_is_better)
            mg.append(np.mean(gaps) if gaps else np.nan)
        ax.plot(range(len(cutoffs)), mg, '-', color='gray', alpha=0.35, linewidth=1)
        ax.annotate(f'k={k}', (len(cutoffs)-1, mg[-1]), fontsize=6,
                    color='gray', alpha=0.6, ha='left', va='center',
                    xytext=(5, 0), textcoords='offset points')

    for mk in adaptive_method_keys:
        mg, clo, chi = [], [], []
        for c in cutoffs:
            _, _, gaps = get_win_and_gaps(results, mk, c, higher_is_better)
            if len(gaps) >= 2:
                m, se = np.mean(gaps), np.std(gaps, ddof=1) / math.sqrt(len(gaps))
                mg.append(m); clo.append(m - 1.96 * se); chi.append(m + 1.96 * se)
            else:
                mg.append(np.nan); clo.append(np.nan); chi.append(np.nan)
        ax.plot(range(len(cutoffs)), mg, 'o-', color=adapt_colors[mk],
                linewidth=2.5, markersize=6, label=adaptive_labels[mk])
        ax.fill_between(range(len(cutoffs)), clo, chi, alpha=0.15, color=adapt_colors[mk])

    # Oracle: dashed gold line with CI
    oracle_mg, oracle_clo, oracle_chi = [], [], []
    for c in cutoffs:
        _, _, gaps = get_win_and_gaps(results, ORACLE_KEY, c, higher_is_better)
        if len(gaps) >= 2:
            m, se = np.mean(gaps), np.std(gaps, ddof=1) / math.sqrt(len(gaps))
            oracle_mg.append(m); oracle_clo.append(m - 1.96 * se); oracle_chi.append(m + 1.96 * se)
        else:
            oracle_mg.append(np.nan); oracle_clo.append(np.nan); oracle_chi.append(np.nan)
    ax.plot(range(len(cutoffs)), oracle_mg, 's--', color='#ff7f00',
            linewidth=2, markersize=6, label='Oracle (best k)')
    ax.fill_between(range(len(cutoffs)), oracle_clo, oracle_chi, alpha=0.1, color='#ff7f00')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(cutoffs)))
    ax.set_xticklabels(cl)
    ax.set_xlabel('Timestep Cutoff')
    ax.set_ylabel('Mean Gap % (positive = GB better)')
    ax.set_title(f'{metric_name}: Mean Gap of GB over DQN+LN (n={len(seeds)})')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(base / f'{safe_name}_meangap.png', dpi=150)
    plt.close()
    print(f'  -> {safe_name}_meangap.png')


# ══════════════════════════════════════════════════
# 1. AUC (already exists, regenerate for consistency)
# ══════════════════════════════════════════════════
print('\n1. AUC of Episode Returns')
auc_results = compute_comparison(compute_auc, higher_is_better=True)
generate_table_and_charts(auc_results, 'AUC of Episode Returns', higher_is_better=True)

# ══════════════════════════════════════════════════
# 2. Final Performance (mean of last 100 episodes)
# ══════════════════════════════════════════════════
print(f'\n2. Final Performance (last {FINAL_WINDOW} episodes)')
final_results = compute_comparison(
    lambda eps, c: compute_final_performance(eps, c, FINAL_WINDOW),
    higher_is_better=True,
)
generate_table_and_charts(final_results, f'Final Performance (last {FINAL_WINDOW} eps)', higher_is_better=True)

# ══════════════════════════════════════════════════
# 3. Best Rolling Mean
# ══════════════════════════════════════════════════
print(f'\n3. Best Rolling Mean (window={ROLLING_WINDOW})')
best_results = compute_comparison(
    lambda eps, c: compute_best_rolling(eps, c, ROLLING_WINDOW),
    higher_is_better=True,
)
generate_table_and_charts(best_results, f'Best Rolling Mean (w={ROLLING_WINDOW})', higher_is_better=True)

# ══════════════════════════════════════════════════
# 4. Time-to-Threshold
# ══════════════════════════════════════════════════
# First figure out a reasonable threshold from LN baseline
print('\n4. Time-to-Threshold')
ln_final_returns = []
for seed in seeds:
    eps = episode_data.get((seed, 'ln'))
    if eps and len(eps) >= FINAL_WINDOW:
        ln_final_returns.append(np.mean([r for _, r in eps[-FINAL_WINDOW:]]))

median_ln = np.median(ln_final_returns)
# Use thresholds at different fractions of median LN final performance
thresholds = [
    round(median_ln * 0.5, 1),
    round(median_ln * 0.7, 1),
    round(median_ln * 0.9, 1),
]
print(f'  LN median final return: {median_ln:.1f}')
print(f'  Thresholds: {thresholds}')

for threshold in thresholds:
    print(f'\n  Time-to-threshold={threshold}')
    ttt_data = compute_ttt_comparison(threshold)

    # Package into cutoff-keyed format (TTT doesn't vary by cutoff, but we
    # report it once using cutoff=200000 to fit the table framework)
    ttt_results = {}
    for (seed, method), val in ttt_data.items():
        for c in cutoffs:
            ttt_results[(seed, method, c)] = val

    # For TTT, we want a simple table: one row per k, columns = win rate + mean difference
    lines = [
        f'Time-to-Threshold (threshold={threshold}, rolling window={ROLLING_WINDOW})',
        f'LN median final return: {median_ln:.1f}',
        f'n = {len(seeds)} seeds',
        '',
    ]

    wins_data = {}
    for k in k_values:
        wins, n, diffs = 0, 0, []
        for s in seeds:
            ln_val = ttt_data.get((s, 'ln'))
            gb_val = ttt_data.get((s, f'gb_k{k}'))
            if ln_val is not None and gb_val is not None:
                n += 1
                if gb_val < ln_val:  # lower = faster = better
                    wins += 1
                diffs.append(ln_val - gb_val)  # positive = GB faster
        wins_data[k] = (wins, n, diffs)

    lines.append(f'{"k":>4}  {"Win Rate":>14}  {"Mean Δ steps":>20}  {"Median Δ steps":>16}')
    lines.append('-' * 70)
    for k in k_values:
        wins, n, diffs = wins_data[k]
        pct, lo, hi = wilson_ci(wins, n)
        m_diff = np.mean(diffs) if diffs else 0
        med_diff = np.median(diffs) if diffs else 0
        se = np.std(diffs, ddof=1) / math.sqrt(len(diffs)) if len(diffs) >= 2 else 0
        lines.append(f'  {k:>2}    {pct:4.0f} [{lo:2.0f},{hi:2.0f}]    {m_diff:+8.0f} ±{se*1.96:6.0f}    {med_diff:+10.0f}')

    safe = f'time_to_threshold_{int(threshold)}'
    with open(base / f'{safe}_table.txt', 'w') as f:
        f.write('\n'.join(lines))
    print(f'    -> {safe}_table.txt')

    # Win rate bar chart for TTT
    fig, ax = plt.subplots(figsize=(8, 5))
    wr_vals = [wilson_ci(*wins_data[k][:2])[0] for k in k_values]
    ci_lo = [wilson_ci(*wins_data[k][:2])[1] for k in k_values]
    ci_hi = [wilson_ci(*wins_data[k][:2])[2] for k in k_values]
    x = np.arange(len(k_values))
    bars = ax.bar(x, wr_vals, color='steelblue', alpha=0.7)
    ax.errorbar(x, wr_vals,
                yerr=[np.array(wr_vals)-np.array(ci_lo), np.array(ci_hi)-np.array(wr_vals)],
                fmt='none', color='black', capsize=3)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.set_ylabel('Win Rate (%) — GB reaches threshold faster')
    ax.set_title(f'Time-to-Threshold={threshold}: GB vs DQN+LN (n={len(seeds)})')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(base / f'{safe}_winrate.png', dpi=150)
    plt.close()
    print(f'    -> {safe}_winrate.png')

print('\nAll done!')
