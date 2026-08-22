#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV = Path('results/z2_scan/z2_prod_L64_128_192_256_200_t8_nextpassv1_finishfull_20260324_193047/z2_L64_128_192_256_prod_200_t8.csv')
OUT = Path('article-deliverables/fig3_retry_20260330/fig3_final_production_log10_retry.png')


def main() -> None:
    df = pd.read_csv(CSV)
    df['L'] = pd.to_numeric(df['L'], errors='coerce').astype('Int64')
    df['g'] = pd.to_numeric(df['g'], errors='coerce')
    df['z2_plus_one'] = pd.to_numeric(df['z2_plus_one'], errors='coerce')

    bad = (~np.isfinite(df['z2_plus_one']))
    if 'nan_detected' in df.columns:
        bad = bad | df['nan_detected'].fillna(False).astype(bool)

    plot_df = df[(~bad) & np.isfinite(df['g']) & (df['g'] > 0)].copy()
    plot_df['log10g'] = np.log10(plot_df['g'])

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 10,
        'axes.labelsize': 11,
        'legend.fontsize': 8,
        'axes.linewidth': 0.8,
    })

    fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=500)

    colors = ['#440154', '#3b528b', '#21918c', '#5ec962']
    for i, L in enumerate(sorted(plot_df['L'].dropna().unique())):
        dL = plot_df[plot_df['L'] == L].sort_values('log10g')
        ax.plot(
            dL['log10g'].to_numpy(),
            dL['z2_plus_one'].to_numpy(),
            color=colors[i % len(colors)],
            lw=0.9,
            marker='o',
            ms=1.2,
            alpha=0.95,
            label=f'L={int(L)}'
        )

    ax.axhline(1.25, color='#f39c12', ls='--', lw=0.8, alpha=0.9, label='1.25')
    ax.axhline(2.0, color='#7cb342', ls='--', lw=0.8, alpha=0.9, label='2.0')

    # Match article placeholder scale style: compact, centered transition region.
    ax.set_xlim(-1.55, 1.05)
    ax.set_ylim(1.28, 2.01)
    ax.set_xlabel(r'$\log g$')
    ax.set_ylabel(r'$1+\langle z^2 \rangle$')
    ax.grid(True, alpha=0.22, lw=0.5)
    ax.legend(loc='lower right', frameon=False, ncol=2, handlelength=1.5)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches='tight')
    print(f'Saved {OUT}')


if __name__ == '__main__':
    main()
