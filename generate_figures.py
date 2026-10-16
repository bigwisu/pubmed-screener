"""
Generate publication-quality figures for JBI manuscript.
Uses Okabe-Ito colorblind-safe palette, clean typography, and exact data from SQLite/tables.
"""

import os
import json
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Colorblind-safe Okabe-Ito palette
OKABE_ITO = {
    'black': '#000000',
    'orange': '#E69F00',
    'sky_blue': '#56B4E9',
    'bluish_green': '#009E73',
    'yellow': '#F0E442',
    'blue': '#0072B2',
    'vermillion': '#D55E00',
    'reddish_purple': '#CC79A7',
    'gray': '#7F7F7F',
    'light_gray': '#D3D3D3'
}

def set_pub_style():
    plt.rcParams.update({
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.family': 'sans-serif',
        'font.size': 9,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'axes.titleweight': 'bold',
        'xtick.labelsize': 8.5,
        'ytick.labelsize': 8.5,
        'legend.fontsize': 8.5,
        'figure.titlesize': 11,
        'lines.linewidth': 1.5,
        'axes.linewidth': 0.8,
        'grid.linewidth': 0.5,
        'grid.alpha': 0.3,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05
    })

def generate_figure_study_overview(out_dir):
    """Figure 1: Schematic of the Decision Model Architecture and Dual Evaluation Framework."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    ax1.axis('off')
    ax2.axis('off')

    # Panel A: Model Architecture
    ax1.set_title('A. Model Architecture & Inference Pipeline', loc='left', fontsize=10, fontweight='bold')
    
    # Input box
    ax1.text(0.5, 0.88, 'Systematic Review PICO Objective\n+ Candidate Title & Abstract',
             ha='center', va='center', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f4f8', edgecolor=OKABE_ITO['blue'], lw=1.2))
    
    ax1.annotate('', xy=(0.5, 0.68), xytext=(0.5, 0.77),
                 arrowprops=dict(arrowstyle='->', lw=1.2, color=OKABE_ITO['black']))
    
    # Encoder Box
    ax1.text(0.5, 0.58, 'General-Domain Contextual Encoder\n(ModernBERT-large, 395M params)',
             ha='center', va='center', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#eef8f2', edgecolor=OKABE_ITO['bluish_green'], lw=1.2))
    
    ax1.annotate('', xy=(0.5, 0.38), xytext=(0.5, 0.47),
                 arrowprops=dict(arrowstyle='->', lw=1.2, color=OKABE_ITO['black']))
    
    # Head Box
    ax1.text(0.5, 0.28, 'Task-Adapted Decision Head\n(Trained on 24,624 SYNERGY pairs)',
             ha='center', va='center', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff5eb', edgecolor=OKABE_ITO['orange'], lw=1.2))
    
    ax1.annotate('', xy=(0.5, 0.08), xytext=(0.5, 0.17),
                 arrowprops=dict(arrowstyle='->', lw=1.2, color=OKABE_ITO['black']))
    
    # Output decision
    ax1.text(0.5, 0.02, r'Score $s \in [0, 1] \geq \theta^* (0.028) \rightarrow \mathbf{Include}$',
             ha='center', va='center', fontsize=8.5, fontweight='bold', color=OKABE_ITO['blue'],
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f7f8fa', edgecolor=OKABE_ITO['gray'], lw=1))

    # Panel B: Dual Evaluation Framework
    ax2.set_title('B. Dual Evaluation Benchmark Designs', loc='left', fontsize=10, fontweight='bold')
    
    # Calibration Box
    ax2.text(0.5, 0.88, r'Calibration Cohort ($N=4$ SRs, 62,120 evaluations)' + '\n' + r'Derived frozen threshold $\theta^* = 0.028$',
             ha='center', va='center', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff8e6', edgecolor=OKABE_ITO['yellow'], lw=1.2))
    
    ax2.annotate('', xy=(0.5, 0.72), xytext=(0.5, 0.80),
                 arrowprops=dict(arrowstyle='->', lw=1.2, color=OKABE_ITO['black']))

    # Primary Benchmark Box
    ax2.text(0.5, 0.65, r'Primary Test Cohort ($N=62$ Unseen SRs, 2,431 Inclusions)',
             ha='center', va='center', fontsize=8.5, fontweight='bold', color=OKABE_ITO['black'])

    # Cross Topic Box
    ax2.text(0.25, 0.35, 'Design 1: Cross-Topic\n' + r'• 962,860 doc-review pairs' + '\n' + r'• Prevalence: 0.25%' + '\n' + r'• Tests extreme ranking' + '\n' + r'• Recall: 95.15%' + '\n' + r'• Specificity: 94.86%',
             ha='center', va='center', fontsize=7.5,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f4f8', edgecolor=OKABE_ITO['sky_blue'], lw=1))

    # Boolean Search Box
    ax2.text(0.75, 0.35, 'Design 2: Boolean-Search\n' + r'• 16,598 candidate pairs' + '\n' + r'• Prevalence: 16.4%' + '\n' + r'• Operational deployment' + '\n' + r'• Recall: 95.15% (identical)' + '\n' + r'• Workload: 77.6% (WSS: 10.4%)',
             ha='center', va='center', fontsize=7.5,
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#eef8f2', edgecolor=OKABE_ITO['bluish_green'], lw=1))

    ax2.annotate('', xy=(0.25, 0.48), xytext=(0.45, 0.58),
                 arrowprops=dict(arrowstyle='->', lw=1, color=OKABE_ITO['gray']))
    ax2.annotate('', xy=(0.75, 0.48), xytext=(0.55, 0.58),
                 arrowprops=dict(arrowstyle='->', lw=1, color=OKABE_ITO['gray']))

    ax2.set_xlim(0, 1)
    ax2.set_ylim(-0.05, 1.0)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.05, 1.0)

    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'figure1_study_overview.pdf'))
    fig.savefig(os.path.join(out_dir, 'figure1_study_overview.png'))
    plt.close(fig)
    print("Generated Figure 1.")

def generate_figure_operating_curve(db_path, out_dir):
    """Figure 2: Operating curves and Threshold sensitivity across 62 primary SRs."""
    conn = sqlite3.connect(db_path)
    calib_srs = {22226047, 22323502, 22422870, 22777524}
    
    # Load all results
    query = """
    SELECT sr_pmid, pmid, relevance_score, is_inclusion
    FROM screening_results
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df_primary = df[~df['sr_pmid'].isin(calib_srs)].copy()
    
    thresholds = np.linspace(0.020, 0.035, 61)
    recalls = []
    specs = []
    workloads = []
    
    total_pos = df_primary['is_inclusion'].sum()
    total_neg = len(df_primary) - total_pos
    total_docs = len(df_primary)
    
    for th in thresholds:
        tp = ((df_primary['relevance_score'] >= th) & (df_primary['is_inclusion'] == 1)).sum()
        fp = ((df_primary['relevance_score'] >= th) & (df_primary['is_inclusion'] == 0)).sum()
        tn = ((df_primary['relevance_score'] < th) & (df_primary['is_inclusion'] == 0)).sum()
        
        recall = tp / total_pos if total_pos > 0 else 0
        spec = tn / total_neg if total_neg > 0 else 0
        workload = (tp + fp) / total_docs
        
        recalls.append(recall * 100)
        specs.append(spec * 100)
        workloads.append(workload * 100)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    
    # Panel A: Threshold vs Metric
    ax1.plot(thresholds, recalls, label='Recall (Safety)', color=OKABE_ITO['blue'], lw=2)
    ax1.plot(thresholds, specs, label='Cross-Topic Specificity', color=OKABE_ITO['bluish_green'], lw=1.8, ls='--')
    ax1.plot(thresholds, workloads, label='Cross-Topic Workload', color=OKABE_ITO['vermillion'], lw=1.8, ls=':')
    
    ax1.axvline(0.028, color=OKABE_ITO['black'], ls='-', lw=1, alpha=0.7)
    ax1.text(0.0282, 45, r'$\theta^* = 0.028$', color=OKABE_ITO['black'], fontsize=8.5, fontweight='bold')
    
    ax1.set_xlabel('Decision Threshold ($\theta$)')
    ax1.set_ylabel('Metric Value (%)')
    ax1.set_title('A. Operational Threshold Sensitivity', loc='left')
    ax1.set_xlim(0.020, 0.035)
    ax1.set_ylim(0, 102)
    ax1.legend(loc='lower left', frameon=False, fontsize=8)
    ax1.grid(True, ls=':', alpha=0.4)
    sns.despine(ax=ax1)
    
    # Panel B: Per-SR Recall Distribution at theta* = 0.028
    sr_perf = df_primary.groupby('sr_pmid').apply(
        lambda g: (g[g['is_inclusion'] == 1]['relevance_score'] >= 0.028).sum() / g['is_inclusion'].sum() * 100
    ).reset_index(name='recall')
    
    bins = np.linspace(0, 100, 21)
    n, bins, patches = ax2.hist(sr_perf['recall'], bins=bins, color=OKABE_ITO['sky_blue'], edgecolor='white', lw=0.8)
    
    # Highlight >=95% bin
    for i, p in enumerate(patches):
        if bins[i] >= 95:
            p.set_facecolor(OKABE_ITO['blue'])
            
    ax2.axvline(95, color=OKABE_ITO['vermillion'], ls='--', lw=1.2, label='95% Target')
    ax2.set_xlabel('Per-Review Recall (%)')
    ax2.set_ylabel('Number of Systematic Reviews')
    ax2.set_title('B. Per-Review Recall Distribution ($N = 62$)', loc='left')
    ax2.text(25, 22, f'≥95% Recall: 45/62 (72.6%)\n100% Recall: 31/62 (50.0%)', 
             fontsize=8, bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7f8fa', edgecolor='#d0d7de'))
    ax2.set_xlim(0, 102)
    ax2.grid(True, ls=':', alpha=0.4, axis='y')
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'figure2_threshold_operating_curves.pdf'))
    fig.savefig(os.path.join(out_dir, 'figure2_threshold_operating_curves.png'))
    plt.close(fig)
    print("Generated Figure 2.")

def generate_figure_noise_floor(db_path, out_dir):
    """Figure 3: Noise floor score distribution and false-negative clustering."""
    conn = sqlite3.connect(db_path)
    calib_srs = {22226047, 22323502, 22422870, 22777524}
    
    query = """
    SELECT sr_pmid, relevance_score, is_inclusion
    FROM screening_results
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    df_primary = df[~df['sr_pmid'].isin(calib_srs)].copy()
    
    # Classes at theta* = 0.028
    th = 0.028
    fn_scores = df_primary[(df_primary['is_inclusion'] == 1) & (df_primary['relevance_score'] < th)]['relevance_score'].values
    tp_scores = df_primary[(df_primary['is_inclusion'] == 1) & (df_primary['relevance_score'] >= th)]['relevance_score'].values
    tn_scores = df_primary[(df_primary['is_inclusion'] == 0) & (df_primary['relevance_score'] < th)]['relevance_score'].sample(n=50000, random_state=42).values
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    
    # Panel A: Boxplot / Violin of scores across prediction categories
    data_plot = [
        pd.DataFrame({'Score': tn_scores, 'Category': 'True Negatives\n(Ineligible)'}),
        pd.DataFrame({'Score': fn_scores, 'Category': 'False Negatives\n(Misses, N=118)'}),
        pd.DataFrame({'Score': tp_scores, 'Category': 'True Positives\n(Included)'})
    ]
    df_box = pd.concat(data_plot, ignore_index=True)
    
    palette = [OKABE_ITO['gray'], OKABE_ITO['vermillion'], OKABE_ITO['blue']]
    sns.boxplot(data=df_box, x='Category', y='Score', hue='Category', palette=palette, legend=False, ax=ax1, width=0.5, fliersize=1.5, linewidth=0.9)
    ax1.axhline(0.028, color=OKABE_ITO['black'], ls='--', lw=1)
    ax1.text(0.5, 0.030, r'$\theta^* = 0.028$', transform=ax1.get_yaxis_transform(), color='black', fontsize=8, fontweight='bold')
    ax1.set_ylabel('Model Relevance Score')
    ax1.set_xlabel('')
    ax1.set_title('A. Score Distributions by Outcome', loc='left')
    ax1.set_yscale('log')
    ax1.grid(True, ls=':', alpha=0.4, axis='y')
    sns.despine(ax=ax1)
    
    # Panel B: High-resolution zoom of FN score band [0.020, 0.030]
    sns.kdeplot(tn_scores, ax=ax2, color=OKABE_ITO['gray'], label='Ineligible Docs (TN)', fill=True, alpha=0.2, lw=1.2, clip=(0.018, 0.032))
    sns.kdeplot(fn_scores, ax=ax2, color=OKABE_ITO['vermillion'], label='Missed Inclusions (FN)', fill=True, alpha=0.4, lw=1.8, clip=(0.018, 0.032))
    ax2.axvline(0.028, color=OKABE_ITO['black'], ls='--', lw=1.2, label=r'Threshold $\theta^*=0.028$')
    
    ax2.text(0.0245, 120, 'Uncertainty Zone\n[0.0203, 0.0279]', fontsize=8, color=OKABE_ITO['vermillion'],
             bbox=dict(boxstyle='round,pad=0.2', facecolor='#fff0f0', edgecolor='#ffc0c0'))
    
    ax2.set_xlim(0.018, 0.032)
    ax2.set_xlabel('Relevance Score (Near Decision Boundary)')
    ax2.set_ylabel('Density')
    ax2.set_title('B. Noise-Floor Decision Boundary Zoom', loc='left')
    ax2.legend(loc='upper left', frameon=False, fontsize=7.5)
    ax2.grid(True, ls=':', alpha=0.4)
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'figure3_noise_floor_distribution.pdf'))
    fig.savefig(os.path.join(out_dir, 'figure3_noise_floor_distribution.png'))
    plt.close(fig)
    print("Generated Figure 3.")

def generate_figure_counterfactual(out_dir):
    """Figure 4: Level 2 Instruction counterfactual experiment across 5 target SRs."""
    # Data from RQ3_PLAN.md and Level 2 Results
    data = [
        {'pmid': '27802478', 'topic': 'Gout management', 'orig_fn': 25, 'alt_fn': 13, 'orig_fp': 659, 'alt_fp': 3736, 'met': False},
        {'pmid': '31585960', 'topic': 'COPD prognostics', 'orig_fn': 16, 'alt_fn': 50, 'orig_fp': 879, 'alt_fp': 1590, 'met': False},
        {'pmid': '31200992', 'topic': 'Mental health/conflict', 'orig_fn': 9, 'alt_fn': 2, 'orig_fp': 371, 'alt_fp': 1155, 'met': True},
        {'pmid': '30409774', 'topic': 'ICI safety', 'orig_fn': 5, 'alt_fn': 6, 'orig_fp': 299, 'alt_fp': 3051, 'met': False},
        {'pmid': '27802505', 'topic': 'Gout diagnosis', 'orig_fn': 3, 'alt_fn': 2, 'orig_fp': 286, 'alt_fp': 1614, 'met': False},
    ]
    df_cf = pd.DataFrame(data)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 3.2))
    
    x = np.arange(len(df_cf))
    width = 0.35
    
    # Panel A: False Negatives Before vs After
    rects1 = ax1.bar(x - width/2, df_cf['orig_fn'], width, label='Original Objective', color=OKABE_ITO['sky_blue'], edgecolor='white')
    rects2 = ax1.bar(x + width/2, df_cf['alt_fn'], width, label='Alternative Objective', color=OKABE_ITO['vermillion'], edgecolor='white')
    
    # Mark the successful one
    ax1.annotate('Vocabulary Gap\n(77.8% Recovered)', xy=(2 + width/2, 2), xytext=(2, 20),
                 arrowprops=dict(facecolor=OKABE_ITO['bluish_green'], arrowstyle='->', lw=1.2),
                 fontsize=7.5, ha='center', color=OKABE_ITO['bluish_green'], fontweight='bold')
    
    ax1.set_ylabel('False Negative Count (Misses)')
    ax1.set_title('A. False Negatives (Misses)', loc='left')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"SR {p}\n({t})" for p, t in zip(df_cf['pmid'], df_cf['topic'])], rotation=25, ha='right', fontsize=7.5)
    ax1.legend(frameon=False, fontsize=8)
    ax1.grid(True, ls=':', alpha=0.4, axis='y')
    sns.despine(ax=ax1)
    
    # Panel B: False Positives (Workload Inflation)
    rects3 = ax2.bar(x - width/2, df_cf['orig_fp'], width, label='Original FP', color=OKABE_ITO['sky_blue'], edgecolor='white')
    rects4 = ax2.bar(x + width/2, df_cf['alt_fp'], width, label='Alternative FP', color=OKABE_ITO['orange'], edgecolor='white')
    
    ax2.set_ylabel('False Positive Count (Extra Workload)')
    ax2.set_title('B. False Positives (Workload Inflation)', loc='left')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"SR {p}\n({t})" for p, t in zip(df_cf['pmid'], df_cf['topic'])], rotation=25, ha='right', fontsize=7.5)
    ax2.legend(frameon=False, fontsize=8)
    ax2.grid(True, ls=':', alpha=0.4, axis='y')
    sns.despine(ax=ax2)
    
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, 'figure4_instruction_counterfactual.pdf'))
    fig.savefig(os.path.join(out_dir, 'figure4_instruction_counterfactual.png'))
    plt.close(fig)
    print("Generated Figure 4.")

def main():
    db_path = 'outputs/soces_pico_laya_e2e_results.sqlite'
    out_dir = 'publication/figures'
    os.makedirs(out_dir, exist_ok=True)
    
    set_pub_style()
    print("Generating manuscript figures...")
    generate_figure_study_overview(out_dir)
    generate_figure_operating_curve(db_path, out_dir)
    generate_figure_noise_floor(db_path, out_dir)
    generate_figure_counterfactual(out_dir)
    print("All figures successfully created in publication/figures/")

if __name__ == '__main__':
    main()
