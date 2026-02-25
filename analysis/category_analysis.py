"""
Phase 2: Category-Level Analysis of Thinking vs Base Models

Reads benchmark_survey.csv and category_tags.json, computes per-category 
deltas (thinking - base), and generates comparison visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import json
import os

matplotlib.use('Agg')

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
FIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

def load_data():
    """Load benchmark survey data and category tags."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'benchmark_survey.csv'))
    with open(os.path.join(DATA_DIR, 'category_tags.json'), 'r') as f:
        categories = json.load(f)
    return df, categories

def compute_deltas(df):
    """Filter to entries with valid deltas and compute summary stats."""
    # Keep only rows with actual scores
    valid = df.dropna(subset=['thinking_score', 'base_score', 'delta'])
    valid = valid[valid['source'] != 'needs_collection']
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK SURVEY: Thinking vs Base Model Comparison")
    print(f"{'='*60}")
    print(f"\nTotal entries: {len(df)}")
    print(f"Entries with valid deltas: {len(valid)}")
    print(f"Entries needing collection: {len(df) - len(valid)}")
    
    return valid

def category_analysis(valid_df):
    """Compute per-category statistics."""
    print(f"\n{'='*60}")
    print(f"PER-CATEGORY DELTA (Thinking - Base)")
    print(f"{'='*60}")
    
    category_stats = valid_df.groupby('category').agg(
        mean_delta=('delta', 'mean'),
        std_delta=('delta', 'std'),
        min_delta=('delta', 'min'),
        max_delta=('delta', 'max'),
        count=('delta', 'count')
    ).sort_values('mean_delta', ascending=True)
    
    for cat, row in category_stats.iterrows():
        sign = '+' if row['mean_delta'] > 0 else ''
        indicator = '✓ HELPS' if row['mean_delta'] > 0 else '✗ HURTS'
        print(f"\n  {cat:25s} | Delta = {sign}{row['mean_delta']:.1f} | "
              f"range [{row['min_delta']:.1f}, {row['max_delta']:.1f}] | "
              f"n={int(row['count'])} | {indicator}")
    
    return category_stats

def model_family_analysis(valid_df):
    """Analyze which model families show the effect most clearly."""
    print(f"\n{'='*60}")
    print(f"PER-MODEL FAMILY DELTA")
    print(f"{'='*60}")
    
    for family in valid_df['model_family'].unique():
        family_df = valid_df[valid_df['model_family'] == family]
        print(f"\n  {family}:")
        for _, row in family_df.iterrows():
            sign = '+' if row['delta'] > 0 else ''
            print(f"    {row['benchmark']:25s} ({row['category']:15s}) "
                  f"| {row['thinking_model']:>8s}: {row['thinking_score']:.1f} "
                  f"| {row['base_model']:>8s}: {row['base_score']:.1f} "
                  f"| Delta = {sign}{row['delta']:.1f}")

def plot_category_heatmap(valid_df):
    """Create a heatmap of thinking-base deltas by category and model family."""
    pivot = valid_df.pivot_table(
        values='delta', 
        index='category', 
        columns='model_family', 
        aggfunc='mean'
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.cm.RdYlGn  # Red=hurts, Green=helps
    
    im = ax.imshow(pivot.values, cmap=cmap, aspect='auto', vmin=-10, vmax=50)
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                color = 'white' if abs(val) > 25 else 'black'
                ax.text(j, i, f'{val:+.1f}', ha='center', va='center', 
                       color=color, fontsize=10, fontweight='bold')
    
    ax.set_title('Thinking − Base Model Performance Delta by Category', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Model Family')
    ax.set_ylabel('Task Category')
    
    plt.colorbar(im, ax=ax, label='Delta Score (Thinking − Base)', shrink=0.8)
    plt.tight_layout()
    
    outpath = os.path.join(FIG_DIR, 'category_delta_heatmap.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outpath}")

def plot_delta_bars(valid_df):
    """Create bar chart of per-benchmark deltas, colored by category."""
    colors = {
        'math': '#2ecc71', 'reasoning': '#3498db', 'code': '#9b59b6',
        'knowledge': '#f39c12', 'commonsense': '#e74c3c',
        'instruction_following': '#e67e22', 'creative': '#1abc9c',
        'safety': '#95a5a6', 'multimodal': '#34495e'
    }
    
    # Sort by delta
    sorted_df = valid_df.sort_values('delta', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(sorted_df) * 0.4)))
    
    labels = [f"{row['benchmark']} ({row['model_family']})" 
              for _, row in sorted_df.iterrows()]
    deltas = sorted_df['delta'].values
    bar_colors = [colors.get(row['category'], '#95a5a6') 
                  for _, row in sorted_df.iterrows()]
    
    bars = ax.barh(range(len(labels)), deltas, color=bar_colors, edgecolor='white')
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    
    ax.axvline(x=0, color='black', linewidth=1.5, linestyle='-')
    ax.set_xlabel('Score Delta (Thinking − Base)', fontsize=12)
    ax.set_title('Does Thinking Help or Hurt?', fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (val, bar) in enumerate(zip(deltas, bars)):
        x_pos = val + 0.5 if val >= 0 else val - 0.5
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, i, f'{val:+.1f}', va='center', ha=ha, fontsize=9)
    
    # Legend for categories
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[k], label=k) 
                      for k in colors if k in sorted_df['category'].values]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8)
    
    plt.tight_layout()
    outpath = os.path.join(FIG_DIR, 'benchmark_delta_bars.png')
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {outpath}")

def identify_thinking_hurts(valid_df):
    """Highlight benchmarks where thinking consistently hurts."""
    hurts = valid_df[valid_df['delta'] < 0]
    
    print(f"\n{'='*60}")
    print(f"BENCHMARKS WHERE THINKING HURTS (Δ < 0)")
    print(f"{'='*60}")
    
    if len(hurts) == 0:
        print("  No negative deltas found in current data.")
        print("  Key gaps to fill: commonsense (HellaSwag, WinoGrande, PIQA),")
        print("  instruction-following (IFEval), creative (AlpacaEval, MT-Bench)")
    else:
        for _, row in hurts.iterrows():
            print(f"  {row['benchmark']:25s} | {row['model_family']:10s} | "
                  f"Δ = {row['delta']:.1f}")
    
    return hurts

def main():
    df, categories = load_data()
    valid_df = compute_deltas(df)
    
    if len(valid_df) == 0:
        print("\nNo valid data to analyze. Fill in benchmark_survey.csv first.")
        return
    
    category_stats = category_analysis(valid_df)
    model_family_analysis(valid_df)
    hurts = identify_thinking_hurts(valid_df)
    
    # Generate plots
    plot_delta_bars(valid_df)
    plot_category_heatmap(valid_df)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"  Categories where thinking helps most:")
    top_helps = category_stats.tail(3)
    for cat, row in top_helps.iterrows():
        print(f"    {cat}: Δ = +{row['mean_delta']:.1f}")
    
    print(f"\n  Categories where thinking potentially hurts (data needed):")
    for cat_name, cat_info in categories['categories'].items():
        if cat_info['hypothesis'] in ('thinking_hurts', 'thinking_neutral_or_hurts'):
            print(f"    {cat_name}: hypothesis = {cat_info['hypothesis']}")
    
    print(f"\n  Key related work to cite:")
    for name, info in categories['related_work'].items():
        print(f"    {name}: {info['finding'][:80]}...")

if __name__ == '__main__':
    main()
