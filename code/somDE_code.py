"""
Analysis of sim_easy.h5ad  –  Spatially Variable Gene Detection via SOMDE
"""

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
from somde import SomNode

adata = sc.read_h5ad('sim_medium.h5ad')

print("=" * 50)
print(f"Shape       : {adata.shape}  (cells x genes)")
print(f"obs columns : {adata.obs.columns.tolist()}")
print(f"var columns : {adata.var.columns.tolist()}")
print(f"obsm keys   : {list(adata.obsm.keys())}")
print(f"uns keys    : {list(adata.uns.keys())}")

if 'ground_truth' in adata.var.columns:
    n_true = adata.var['ground_truth'].sum()
    print(f"\nGround-truth SVGs: {n_true} / {adata.n_vars}")
elif 'is_svg' in adata.var.columns:
    n_true = adata.var['is_svg'].sum()
    print(f"\nGround-truth SVGs (is_svg): {n_true} / {adata.n_vars}")
else:
    print("\nNo ground-truth column found in adata.var")

expr = adata.X
if hasattr(expr, 'toarray'):
    expr = expr.toarray()
expr = np.array(expr, dtype=np.float64)

print("\nExpression stats:")
print(f"  dtype  : {adata.X.dtype}")
print(f"  min    : {expr.min():.4f}")
print(f"  max    : {expr.max():.4f}")
print(f"  mean   : {expr.mean():.4f}")
print(f"  % zeros: {(expr == 0).mean() * 100:.1f}%")

 # If the matrix looks log-transformed (max < 30), revert with expm1.
is_log = expr.max() < 30
print(f"\n  Looks log-transformed: {is_log}")
if is_log:
    expr_counts = np.expm1(expr)
    print("  → Reverted to raw counts via expm1")
else:
    expr_counts = expr

xy = adata.obsm['spatial']

plt.figure(figsize=(6, 5))
plt.scatter(xy[:, 0], xy[:, 1], s=2, alpha=0.4)
plt.xlabel('x'); plt.ylabel('y')
plt.title('sim_medium – cell positions')
plt.tight_layout()
plt.savefig('sim_medium_spatial.png', dpi=150)
plt.show()
print("Saved: sim_medium_spatial.png")

# ── 4. Filter genes with zero total expression ────────────────────────────────
gene_totals = expr_counts.sum(axis=0)
keep = gene_totals > 0
print(f"\nGenes kept (total > 0): {keep.sum()} / {adata.n_vars}")
expr_filt = expr_counts[:, keep]
gene_names = adata.var_names[keep]

# ──  SOMDE ──────────────────────────────────────────────────────────────────
 df_somde = pd.DataFrame(
    expr_filt.T.astype(np.float64),
    index=gene_names,
    columns=adata.obs_names
)

n_cells = adata.n_obs
grid_size = max(5, int(np.sqrt(n_cells / 20)))
print(f"\nSOMDE: {n_cells} cells, grid_size={grid_size}")

som = SomNode(xy.astype(np.float64), grid_size)
ndf, ninfo = som.mtx(df_somde)
_ = som.norm()

result, SVnum = som.run()

print(f"\nFound {SVnum} spatially variable genes  (q < 0.05)")
print("\nTop 10 SVGs:")
print(result.nsmallest(10, 'qval')[['g', 'LLR', 'pval', 'qval']].to_string(index=False))

result.to_csv('sim_medium_somde_results.csv', index=False)
print("\nSaved: sim_medium_somde_results.csv")

gt_col = None
if 'ground_truth' in adata.var.columns:
    gt_col = 'ground_truth'
elif 'is_svg' in adata.var.columns:
    gt_col = 'is_svg'

if gt_col:
    from sklearn.metrics import roc_auc_score, average_precision_score

    # Merge result with ground truth
    gt = adata.var[[gt_col]].copy()
    gt.index.name = 'g'
    merged = result.set_index('g').join(gt, how='inner')

    labels = merged[gt_col].astype(int).values
    # Lower qval = more likely SVG; invert for scoring
    scores = -np.log10(merged['qval'].clip(1e-300, 1).values)

    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    print(f"\nEvaluation vs ground truth:")
    print(f"  AUROC : {auroc:.4f}")
    print(f"  AUPRC : {auprc:.4f}")

# ── Visualise ──────────────────────────────────────────
top_genes = result.nsmallest(6, 'qval')['g'].tolist()

fig, axes = plt.subplots(2, 3, figsize=(12, 8))
axes = axes.flatten()

for ax, gene in zip(axes, top_genes):
    if gene in list(adata.var_names):
        gidx = list(adata.var_names).index(gene)
        vals = expr[:, gidx]
    else:
        vals = np.zeros(adata.n_obs)

    sc_ = ax.scatter(xy[:, 0], xy[:, 1], c=vals, s=2, cmap='viridis')
    plt.colorbar(sc_, ax=ax, shrink=0.8)
    ax.set_title(gene, fontsize=9)
    ax.set_xlabel('x'); ax.set_ylabel('y')

# hide unused axes
for ax in axes[len(top_genes):]:
    ax.set_visible(False)

plt.suptitle('Top 6 SVGs – spatial expression (sim_medium)', fontsize=11)
plt.tight_layout()
plt.savefig('sim_medium_top_svgs.png', dpi=150)
plt.show()
print("Saved: sim_medium_top_svgs.png")

# ── 8. LLR distribution ───────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
plt.hist(result['LLR'], bins=40, edgecolor='k', alpha=0.7)
plt.xlabel('Log-Likelihood Ratio (LLR)')
plt.ylabel('Count')
plt.title('SOMDE LLR distribution – sim_medium')
plt.tight_layout()
plt.savefig('sim_medium_llr_hist.png', dpi=150)
plt.show()
print("Saved: sim_medium_llr_hist.png")
