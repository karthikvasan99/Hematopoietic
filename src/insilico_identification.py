import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Import dataset & model classes from model_fit
from model_fit import MicrobiomeDataset, Phase1_Train, Phase1_Val, Phase2_Dynamics, run_phase_1, run_phase_2, save_model_results

def resolve_paths():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(base_dir, '..'))
    
    if os.path.exists(os.path.join(project_root, 'Data')):
        data_dir = os.path.join(project_root, 'Data')
    elif os.path.exists('./Data'):
        data_dir = './Data'
    else:
        data_dir = '/home/karthik_unix/Hematopoietic/Data'
        
    cache_dir = base_dir
    return data_dir, cache_dir

def load_or_train_all(K=20):
    data_dir, cache_dir = resolve_paths()
    bundle_path = os.path.join(cache_dir, f'model_params_K{K}.pt')
    
    if os.path.exists(bundle_path):
        print(f"Loading existing model parameters from {bundle_path}...")
        params = torch.load(bundle_path, map_location='cpu', weights_only=False)
        return params
        
    print(f"Bundle {bundle_path} not found. Loading data and Phase 1 cache to extract Phase 2...")
    trajectory_df = pd.read_csv(os.path.join(data_dir, 'trajectory_data.csv'))
    asv_df_raw = pd.read_csv(os.path.join(data_dir, 'tblcounts_asv_wide.csv'))
    
    asv_df = asv_df_raw.set_index('ASV').T 
    asv_df = asv_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    rel_abund_df = asv_df.div(asv_df.sum(axis=1), axis=0).fillna(0)
    mean_rel_abund = rel_abund_df.mean(axis=0)
    asv_df = asv_df.loc[:, mean_rel_abund >= 1e-3]
    
    valid_samples = set(asv_df.index)
    trajectory_df = trajectory_df[trajectory_df['SampleID_Y'].isin(valid_samples)]
    
    patient_list = trajectory_df['PatientID'].unique().tolist()
    patient2idx = {pid: idx for idx, pid in enumerate(patient_list)}
    
    unique_drugs = set()
    for history_str in trajectory_df['Prior_Drugs_30d']:
        for drug in json.loads(history_str):
            unique_drugs.add(drug['Factor'])
    drug2idx = {drug: idx for idx, drug in enumerate(sorted(list(unique_drugs)))}
    
    train_dfs, val_dfs = [], []
    for pid, group in trajectory_df.groupby('PatientID'):
        group = group.sort_values('Day_Y').copy()
        group['prev_Day_Y'] = group['Day_Y'].shift(1)
        group['prev_SampleID'] = group['SampleID_Y'].shift(1)
        n_samples = len(group)
        if n_samples < 2:
            train_dfs.append(group)
            continue
        split_idx = max(1, int(n_samples * 0.8))
        train_dfs.append(group.iloc[:split_idx])
        val_dfs.append(group.iloc[split_idx:])
        
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    
    train_dataset = MicrobiomeDataset(train_df, asv_df, patient2idx, drug2idx, max_drugs=20)
    val_dataset = MicrobiomeDataset(val_df, asv_df, patient2idx, drug2idx, max_drugs=20)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    Theta_fixed = run_phase_1(train_loader, val_loader, asv_df.shape[1], K=K, epochs=1500, cache_dir=cache_dir)
    dynamics_model = run_phase_2(
        train_loader, val_loader, 
        len(patient2idx), len(drug2idx), 
        K=K, epochs=1000, cache_dir=cache_dir
    )
    
    save_model_results(
        dynamics_model=dynamics_model,
        Theta=Theta_fixed,
        patient_list=patient_list,
        patient2idx=patient2idx,
        asv_names=asv_df.columns.tolist(),
        drug2idx=drug2idx,
        K=K,
        save_dir=cache_dir
    )
    
    return torch.load(bundle_path, map_location='cpu', weights_only=False)

def run_insilico_experiment(params, samples_per_patient=25, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    mu_p = params['mu_p']           # [num_patients, K]
    sigma = params['sigma']         # [K]
    cov_matrix = params['cov_matrix'] # [K, K]
    Theta = params['Theta']         # [K, num_asvs]
    patient_list = params['patient_list']
    num_patients, K = mu_p.shape
    num_asvs = Theta.shape[1]
    
    print("\n" + "="*65)
    print(f"🧪 IN-SILICO IDENTIFICATION EXPERIMENT")
    print(f"   Cohort: {num_patients:,} Patients | Latent Topics: K={K} | ASVs: {num_asvs}")
    print(f"   Synthetic Samples per Patient: {samples_per_patient}")
    print(f"   Total In-Silico Microbiomes: {num_patients * samples_per_patient:,}")
    print("="*65)
    
    # 1. Generate in-silico latent states Z ~ N(mu_p, diag(sigma^2))
    # Shape: [num_patients, samples_per_patient, K]
    eps = torch.randn(num_patients, samples_per_patient, K)
    Z_syn = mu_p.unsqueeze(1) + eps * sigma.view(1, 1, K)
    
    # Flatten synthetic samples for evaluation
    # Z_flat: [N_total, K], y_true: [N_total]
    N_total = num_patients * samples_per_patient
    Z_flat = Z_syn.view(N_total, K)
    y_true = torch.arange(num_patients).repeat_interleave(samples_per_patient)
    
    # Generate relative abundances in ASV space P = softmax(Z * Theta)
    logits = torch.matmul(Z_flat, Theta)
    P_syn = torch.softmax(logits, dim=1) # [N_total, num_asvs]
    
    # Precompute baseline expected relative abundances: \bar{P}_p = softmax(\mu_p * \Theta)
    mu_logits = torch.matmul(mu_p, Theta)
    P_mean_baselines = torch.softmax(mu_logits, dim=1) # [num_patients, num_asvs]
    
    # -------------------------------------------------------------
    # 2. Maximum Likelihood / Mahalanobis Identification in Latent Space
    # -------------------------------------------------------------
    print("\n[1/3] Computing Latent Space Maximum Likelihood Classification...")
    # Log-likelihood of sample z under patient p's distribution:
    # log N(z | mu_p', diag(sigma^2)) = -0.5 * sum_k ((z_k - mu_p',k)^2 / sigma_k^2) + const
    # We can evaluate this in chunks to conserve memory
    
    batch_size = 5000
    top1_count = 0
    top5_count = 0
    top10_count = 0
    top1pct_count = 0
    ranks_all = []
    
    inv_sigma_sq = 1.0 / (sigma ** 2) # [K]
    
    for i in range(0, N_total, batch_size):
        z_batch = Z_flat[i:i+batch_size] # [B, K]
        y_batch = y_true[i:i+batch_size] # [B]
        
        # Compute negative Mahalanobis distance to all patients
        # (z_i - mu_j)^2 / sigma^2 = (z_i^2 - 2 z_i mu_j + mu_j^2) / sigma^2
        # score shape: [B, num_patients]
        diff = z_batch.unsqueeze(1) - mu_p.unsqueeze(0) # [B, num_patients, K]
        mahal_sq = (diff ** 2 * inv_sigma_sq.view(1, 1, K)).sum(dim=-1) # [B, num_patients]
        
        # Smallest distance = highest likelihood
        # Sort in ascending order of distance
        sorted_indices = torch.argsort(mahal_sq, dim=1) # [B, num_patients]
        
        # Find the rank of the true patient ID (0-indexed)
        # Find where sorted_indices == y_batch[:, None]
        matches = (sorted_indices == y_batch.unsqueeze(1))
        true_ranks = torch.nonzero(matches)[:, 1] # [B] (rank 0 = top-1)
        
        ranks_all.extend(true_ranks.tolist())
        top1_count += (true_ranks == 0).sum().item()
        top5_count += (true_ranks < 5).sum().item()
        top10_count += (true_ranks < 10).sum().item()
        top1pct_cutoff = max(1, int(num_patients * 0.01))
        top1pct_count += (true_ranks < top1pct_cutoff).sum().item()
        
    ranks_all = np.array(ranks_all)
    random_chance_top1 = 1.0 / num_patients
    random_chance_top5 = 5.0 / num_patients
    
    top1_acc = (top1_count / N_total) * 100
    top5_acc = (top5_count / N_total) * 100
    top10_acc = (top10_count / N_total) * 100
    top1pct_acc = (top1pct_count / N_total) * 100
    median_rank = np.median(ranks_all) + 1
    mean_rank = np.mean(ranks_all) + 1
    median_percentile = (median_rank / num_patients) * 100
    
    print("\n--- Latent Space Identification Results ---")
    print(f"  • Top-1 Accuracy:       {top1_acc:.2f}%  (Random chance: {random_chance_top1*100:.3f}%)")
    print(f"  • Top-5 Accuracy:       {top5_acc:.2f}%  (Random chance: {random_chance_top5*100:.3f}%)")
    print(f"  • Top-10 Accuracy:      {top10_acc:.2f}%")
    print(f"  • Top-1% Accuracy:      {top1pct_acc:.2f}% (Top {max(1, int(num_patients * 0.01))} candidates)")
    print(f"  • Median Rank of True:  {median_rank:.1f} / {num_patients} (Top {median_percentile:.3f}% percentile)")
    print(f"  • Mean Rank of True:    {mean_rank:.1f} / {num_patients}")
    
    # -------------------------------------------------------------
    # 3. Relative Abundance (ASV) Space Classification (Cosine / Dot-Product)
    # -------------------------------------------------------------
    print("\n[2/3] Computing Relative Abundance (ASV) Space Identification...")
    # Evaluate nearest patient profile in compositional relative abundance space
    # Using Cosine Similarity between P_syn and baseline P_mean_baselines
    
    asv_top1, asv_top5 = 0, 0
    asv_ranks = []
    
    # Normalize profiles for cosine similarity
    P_syn_norm = F.normalize(P_syn, p=2, dim=1)
    P_base_norm = F.normalize(P_mean_baselines, p=2, dim=1)
    
    for i in range(0, N_total, batch_size):
        p_batch = P_syn_norm[i:i+batch_size]
        y_batch = y_true[i:i+batch_size]
        
        sims = torch.matmul(p_batch, P_base_norm.t()) # [B, num_patients]
        sorted_indices = torch.argsort(sims, dim=1, descending=True)
        
        matches = (sorted_indices == y_batch.unsqueeze(1))
        true_ranks = torch.nonzero(matches)[:, 1]
        
        asv_ranks.extend(true_ranks.tolist())
        asv_top1 += (true_ranks == 0).sum().item()
        asv_top5 += (true_ranks < 5).sum().item()
        
    asv_ranks = np.array(asv_ranks)
    asv_top1_acc = (asv_top1 / N_total) * 100
    asv_top5_acc = (asv_top5 / N_total) * 100
    
    print("\n--- ASV Compositional Space Identification Results ---")
    print(f"  • Top-1 Accuracy:       {asv_top1_acc:.2f}%  (Random chance: {random_chance_top1*100:.3f}%)")
    print(f"  • Top-5 Accuracy:       {asv_top5_acc:.2f}%  (Random chance: {random_chance_top5*100:.3f}%)")
    print(f"  • Median Rank of True:  {np.median(asv_ranks)+1:.1f} / {num_patients}")

    # -------------------------------------------------------------
    # 4. Intraclass Correlation Coefficient (ICC) Across Latent Dimensions
    # -------------------------------------------------------------
    print("\n[3/3] Calculating Intraclass Correlation Coefficients (ICC)...")
    # ICC_k = Var(mu_p,k) / (Var(mu_p,k) + sigma_k^2)
    var_between = torch.var(mu_p, dim=0).numpy() # [K]
    var_within = (sigma ** 2).numpy()            # [K]
    icc = var_between / (var_between + var_within + 1e-8)
    
    print("\n--- Latent Dimension Variance Partitioning ---")
    print(f"  • Mean ICC across {K} topics:   {np.mean(icc):.4f} ± {np.std(icc):.4f}")
    print(f"  • Median ICC:                  {np.median(icc):.4f}")
    print(f"  • Range (Min – Max ICC):       [{np.min(icc):.4f}, {np.max(icc):.4f}]")
    print(f"  • Proportion of Dimensions with ICC > 0.50: {np.mean(icc > 0.50)*100:.1f}%")
    
    # -------------------------------------------------------------
    # 5. Plotting and Visualizations
    # -------------------------------------------------------------
    plot_results(mu_p, Z_syn, ranks_all, asv_ranks, icc, num_patients, K)
    
    return {
        'latent_top1': top1_acc,
        'latent_top5': top5_acc,
        'latent_top10': top10_acc,
        'latent_median_rank': median_rank,
        'asv_top1': asv_top1_acc,
        'asv_top5': asv_top5_acc,
        'mean_icc': np.mean(icc),
        'median_icc': np.median(icc),
        'icc_per_dim': icc
    }

def plot_results(mu_p, Z_syn, ranks_latent, ranks_asv, icc, num_patients, K, output_path='insilico_identification_results.png'):
    fig = plt.figure(figsize=(16, 12))
    
    # Subplot 1: PCA Projection of In-Silico Samples for 10 Representative Patients
    ax1 = fig.add_subplot(2, 2, 1)
    num_viz_patients = 10
    selected_pids = np.random.choice(num_patients, num_viz_patients, replace=False)
    
    pca = PCA(n_components=2)
    all_mu_np = mu_p.numpy()
    pca.fit(all_mu_np)
    
    colors = plt.cm.tab10(np.linspace(0, 1, num_viz_patients))
    for i, pid in enumerate(selected_pids):
        z_pts = Z_syn[pid].numpy() # [samples, K]
        z_2d = pca.transform(z_pts)
        mu_2d = pca.transform(all_mu_np[pid:pid+1])
        
        ax1.scatter(z_2d[:, 0], z_2d[:, 1], color=colors[i], alpha=0.6, s=25, label=f'Patient {pid}')
        ax1.scatter(mu_2d[:, 0], mu_2d[:, 1], color=colors[i], edgecolors='black', s=120, marker='*', linewidth=1.5)
        
    ax1.set_title(f'PCA of In-Silico Microbiome Samples (10 Patients)\nStars = True Baseline Centers ($\\mu_p$)', fontsize=13)
    ax1.set_xlabel('Principal Component 1', fontsize=11)
    ax1.set_ylabel('Principal Component 2', fontsize=11)
    ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Cumulative Match Characteristic (CMC) Curve / Top-K Identification
    ax2 = fig.add_subplot(2, 2, 2)
    max_k = 50
    k_vals = np.arange(1, max_k + 1)
    cmc_latent = [(np.sum(ranks_latent < k) / len(ranks_latent)) * 100 for k in k_vals]
    cmc_asv = [(np.sum(ranks_asv < k) / len(ranks_asv)) * 100 for k in k_vals]
    cmc_random = [(k / num_patients) * 100 for k in k_vals]
    
    ax2.plot(k_vals, cmc_latent, 'b-o', markersize=4, label='Latent Space (Mahalanobis / Likelihood)', linewidth=2)
    ax2.plot(k_vals, cmc_asv, 'g--s', markersize=4, label='ASV Compositional Space (Cosine)', linewidth=2)
    ax2.plot(k_vals, cmc_random, 'r:', label=f'Random Chance (1/{num_patients})', linewidth=1.5)
    
    ax2.set_title('Cumulative Match Characteristic (Top-K Accuracy)', fontsize=13)
    ax2.set_xlabel('Top-K Candidates Allowed', fontsize=11)
    ax2.set_ylabel('Identification Accuracy (%)', fontsize=11)
    ax2.set_xlim(1, max_k)
    ax2.set_ylim(0, 102)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Rank Distribution Histogram (Log Scale)
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.hist(ranks_latent + 1, bins=np.logspace(0, np.log10(num_patients), 40), color='royalblue', edgecolor='black', alpha=0.8)
    ax3.set_xscale('log')
    ax3.set_title(f'Distribution of True Patient Rank (Out of {num_patients:,} Patients)', fontsize=13)
    ax3.set_xlabel('True Patient Rank (1 = Exact Top-1 Match)', fontsize=11)
    ax3.set_ylabel('Count of In-Silico Samples', fontsize=11)
    ax3.axvline(np.median(ranks_latent) + 1, color='red', linestyle='--', linewidth=2, label=f'Median Rank = {np.median(ranks_latent)+1:.1f}')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Intraclass Correlation Coefficient (ICC) by Latent Topic
    ax4 = fig.add_subplot(2, 2, 4)
    topic_indices = np.arange(K)
    bars = ax4.bar(topic_indices, icc, color='teal', edgecolor='black', alpha=0.85)
    ax4.axhline(0.5, color='orange', linestyle='--', label='Substantial Personalization (ICC > 0.50)')
    ax4.axhline(np.mean(icc), color='red', linestyle='-', linewidth=2, label=f'Mean ICC = {np.mean(icc):.3f}')
    
    ax4.set_title('Intraclass Correlation (ICC) per Latent Topic\n$\\mathrm{ICC}_k = \\mathrm{Var}(\\mu_p) / [\\mathrm{Var}(\\mu_p) + \\sigma^2]$', fontsize=13)
    ax4.set_xlabel('Latent Topic Index (k)', fontsize=11)
    ax4.set_ylabel('Intraclass Correlation Coefficient', fontsize=11)
    ax4.set_xticks(topic_indices)
    ax4.set_ylim(0, 1.05)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n📊 Summary visualization saved to {output_path}")

if __name__ == '__main__':
    params = load_or_train_all(K=20)
    results = run_insilico_experiment(params, samples_per_patient=25, seed=42)
