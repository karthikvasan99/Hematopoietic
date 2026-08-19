import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

class MicrobiomeDataset(Dataset):
    def __init__(self, df, asv_df, patient2idx, drug2idx, max_drugs=20):
        self.df = df.reset_index(drop=True)
        self.asv_df = asv_df
        self.patient2idx = patient2idx
        self.drug2idx = drug2idx
        self.max_drugs = max_drugs
        
        self.P_targets = []
        for sample_id in self.df['SampleID_Y']:
            counts = self.asv_df.loc[sample_id].values.astype(np.float32)
            P = counts / (counts.sum() + 1e-8)
            self.P_targets.append(P)
            
        self.Z_targets = None 
        self.Z_dict = None 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        patient_idx = self.patient2idx[row['PatientID']]
        day_y = float(row['Day_Y'])
        prior_drugs = json.loads(row['Prior_Drugs_30d'])
        
        has_prev = 0.0
        delta_t_prev = 0.0
        
        if pd.notna(row['prev_Day_Y']):
            has_prev = 1.0
            delta_t_prev = max(0.0, day_y - float(row['prev_Day_Y']))
            
        drug_indices, time_deltas, drug_mask = [], [], []
        
        for drug in prior_drugs:
            if has_prev > 0 and float(drug['StartDayRelativeToNearestHCT']) <= float(row['prev_Day_Y']):
                continue
                
            if len(drug_indices) >= self.max_drugs:
                break 
                
            drug_indices.append(self.drug2idx[drug['Factor']])
            time_deltas.append(max(0.0, day_y - float(drug['StartDayRelativeToNearestHCT'])))
            drug_mask.append(1.0)
            
        while len(drug_indices) < self.max_drugs:
            drug_indices.append(0)
            time_deltas.append(0.0)
            drug_mask.append(0.0) 
            
        batch = {
            'sample_idx': torch.tensor(idx, dtype=torch.long),
            'patient_idx': torch.tensor(patient_idx, dtype=torch.long),
            'drug_indices': torch.tensor(drug_indices, dtype=torch.long),
            'time_deltas': torch.tensor(time_deltas, dtype=torch.float32),
            'drug_mask': torch.tensor(drug_mask, dtype=torch.float32),
            'P_target': torch.tensor(self.P_targets[idx], dtype=torch.float32),
            'has_prev': torch.tensor(has_prev, dtype=torch.float32),
            'delta_t_prev': torch.tensor(delta_t_prev, dtype=torch.float32)
        }
        
        if self.Z_targets is not None:
            batch['Z_target'] = self.Z_targets[idx]
            
        if self.Z_dict is not None and has_prev > 0:
            batch['Z_prev'] = self.Z_dict[row['prev_SampleID']]
        else:
            batch['Z_prev'] = torch.zeros(self.Z_targets.shape[1] if self.Z_targets is not None else 10)
            
        return batch

class Phase1_Train(nn.Module):
    def __init__(self, num_samples, num_asvs, K):
        super().__init__()
        self.Z = nn.Embedding(num_samples, K)
        self.Z.weight.data.normal_(0, 0.1)
        self.Theta = nn.Parameter(torch.randn(K, num_asvs) * 0.1)
        
    def forward(self, sample_idx):
        logits = torch.matmul(self.Z(sample_idx), self.Theta)
        return F.log_softmax(logits, dim=1)

class Phase1_Val(nn.Module):
    def __init__(self, num_samples, Theta_fixed, K):
        super().__init__()
        self.Z = nn.Embedding(num_samples, K)
        self.Z.weight.data.normal_(0, 0.1)
        self.Theta = Theta_fixed 
        
    def forward(self, sample_idx):
        logits = torch.matmul(self.Z(sample_idx), self.Theta)
        return F.log_softmax(logits, dim=1)

class Phase2_Dynamics(nn.Module):
    def __init__(self, num_patients, num_drugs, K):
        super().__init__()
        self.mu_p = nn.Embedding(num_patients, K)
        # Use a global variance parameter rather than patient-specific
        # This completely prevents variance collapse on sparse patients!
        self.log_sigma = nn.Parameter(torch.zeros(K)) 
        self.V_m = nn.Embedding(num_drugs, K)
        
        self.mu_p.weight.data.normal_(0, 0.1)
        self.V_m.weight.data.normal_(0, 0.1)
        self.log_tau = nn.Parameter(torch.tensor(2.3)) # ~10 days
        
    def forward(self, patient_idx, Z_prev, delta_t_prev, has_prev, drug_indices, time_deltas, drug_mask):
        tau = torch.exp(self.log_tau)
        mu = self.mu_p(patient_idx)
        sigma = torch.exp(self.log_sigma)
        
        V_batch = self.V_m(drug_indices)
        decay = torch.exp(-time_deltas / tau)
        pert_batch = V_batch * decay.unsqueeze(-1) * drug_mask.unsqueeze(-1)
        total_pert = pert_batch.sum(dim=1)
        
        decay_prev = torch.exp(-delta_t_prev / tau)
        baseline_term = (Z_prev - mu) * decay_prev.unsqueeze(-1) * has_prev.unsqueeze(-1)
        
        Z_hat = mu + baseline_term + total_pert
        return Z_hat, sigma

def run_phase_1(train_loader, val_loader, num_asvs, K=10, epochs=1500, cache_dir='.'):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Cache File Paths (tied to K to prevent dimension mismatches)
    cache_theta = os.path.join(cache_dir, f'theta_K{K}.pt')
    cache_z_train = os.path.join(cache_dir, f'z_train_K{K}.pt')
    cache_z_val = os.path.join(cache_dir, f'z_val_K{K}.pt')
    cache_z_dict = os.path.join(cache_dir, f'z_dict_K{K}.pt')

    # 1. Cache Check
    if os.path.exists(cache_theta) and os.path.exists(cache_z_dict):
        print(f"\n=== PHASE 1: LOADING FROM CACHE (K={K}) ===")
        print("Found existing Phase 1 tensors. Skipping matrix factorization!")
        Theta_fixed = torch.load(cache_theta, map_location='cpu', weights_only=True)
        train_loader.dataset.Z_targets = torch.load(cache_z_train, map_location='cpu', weights_only=True)
        val_loader.dataset.Z_targets = torch.load(cache_z_val, map_location='cpu', weights_only=True)
        Z_dict = torch.load(cache_z_dict, map_location='cpu', weights_only=False)
        
        train_loader.dataset.Z_dict = Z_dict
        val_loader.dataset.Z_dict = Z_dict
        return Theta_fixed.cpu()

    # 2. Factorize Training Set
    print(f"\n=== PHASE 1: REPRESENTATION LEARNING (on {device}) ===")
    model_train = Phase1_Train(len(train_loader.dataset), num_asvs, K).to(device)
    optimizer = torch.optim.Adam(model_train.parameters(), lr=0.01)
    
    print("Factorizing Training Set...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        grad_z_sum = 0.0
        theta_grad_accum = torch.zeros_like(model_train.Theta)
        
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            log_P_hat = model_train(batch['sample_idx'])
            loss = F.kl_div(log_P_hat, batch['P_target'], reduction='batchmean')
            loss.backward()
            
            batch_size = len(batch['sample_idx'])
            with torch.no_grad():
                grad_z_sum += (model_train.Z.weight.grad[batch['sample_idx']] * batch_size).abs().sum().item()
                theta_grad_accum += model_train.Theta.grad * batch_size
                
            optimizer.step()
            epoch_loss += loss.item() * batch_size
            
        with torch.no_grad():
            N = len(train_loader.dataset)
            weight_z_sum = model_train.Z.weight.abs().sum().item() + 1e-8
            weight_theta_sum = model_train.Theta.abs().sum().item() + 1e-8
            
            z_ratio = grad_z_sum / weight_z_sum
            dataset_mean_grad_theta = theta_grad_accum / N
            theta_ratio = dataset_mean_grad_theta.abs().sum().item() / weight_theta_sum
            conv_metric = z_ratio + theta_ratio
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Train KL: {epoch_loss / N:.4f} | Conv Metric: {conv_metric:.4f}")
            
        if epoch > 10 and conv_metric < 1e-2:
            print(f"Phase 1 Training converged early at Epoch {epoch:04d} (Metric: {conv_metric:.4f} < 0.01)")
            break
            
    Theta_fixed = model_train.Theta.detach()
    train_loader.dataset.Z_targets = model_train.Z.weight.detach().cpu()
    
    # 3. Project Validation Set
    model_val = Phase1_Val(len(val_loader.dataset), Theta_fixed, K).to(device)
    opt_val = torch.optim.Adam(model_val.parameters(), lr=0.01)
    
    print("\nProjecting Validation Set (Frozen Theta)...")
    for epoch in range(epochs):
        epoch_loss = 0.0
        grad_z_sum = 0.0
        
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            opt_val.zero_grad()
            log_P_hat = model_val(batch['sample_idx'])
            loss = F.kl_div(log_P_hat, batch['P_target'], reduction='batchmean')
            loss.backward()
            
            batch_size = len(batch['sample_idx'])
            with torch.no_grad():
                grad_z_sum += (model_val.Z.weight.grad[batch['sample_idx']] * batch_size).abs().sum().item()
                
            opt_val.step()
            epoch_loss += loss.item() * batch_size
            
        with torch.no_grad():
            N = len(val_loader.dataset)
            weight_z_sum = model_val.Z.weight.abs().sum().item() + 1e-8
            conv_metric = grad_z_sum / weight_z_sum
            
        if epoch % 100 == 0:
            print(f"Epoch {epoch:04d} | Val KL: {epoch_loss / N:.4f} | Conv Metric: {conv_metric:.4f}")
            
        if epoch > 10 and conv_metric < 1e-2:
            print(f"Phase 1 Validation converged early at Epoch {epoch:04d} (Metric: {conv_metric:.4f} < 0.01)")
            break
            
    val_loader.dataset.Z_targets = model_val.Z.weight.detach().cpu()
        
    # 4. Construct Temporal Dictionary
    print("\nLinking temporal Z states for Phase 2 and Saving Cache...")
    Z_dict = {}
    for idx, sample_id in enumerate(train_loader.dataset.df['SampleID_Y']):
        Z_dict[sample_id] = train_loader.dataset.Z_targets[idx]
    for idx, sample_id in enumerate(val_loader.dataset.df['SampleID_Y']):
        Z_dict[sample_id] = val_loader.dataset.Z_targets[idx]
        
    train_loader.dataset.Z_dict = Z_dict
    val_loader.dataset.Z_dict = Z_dict

    # 5. Save everything to disk
    torch.save(Theta_fixed.cpu(), cache_theta)
    torch.save(train_loader.dataset.Z_targets, cache_z_train)
    torch.save(val_loader.dataset.Z_targets, cache_z_val)
    torch.save(Z_dict, cache_z_dict)
    
    return Theta_fixed.cpu()

def run_phase_2(train_loader, val_loader, num_patients, num_drugs, K=10, epochs=1000, cache_dir='.'):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model = Phase2_Dynamics(num_patients, num_drugs, K).to(device)
    cache_dynamics = os.path.join(cache_dir, f'dynamics_model_K{K}.pt')
    
    if os.path.exists(cache_dynamics):
        print(f"\n=== PHASE 2: LOADING FROM CACHE (K={K}) ===")
        print("Found existing Phase 2 model weights. Skipping Phase 2 training!")
        model.load_state_dict(torch.load(cache_dynamics, map_location=device, weights_only=True))
        return model

    print(f"\n=== PHASE 2: DYNAMIC MODELING (on {device}) ===")
    
    # Added L2 regularization (weight_decay) for stable drug embeddings
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            optimizer.zero_grad()
            
            Z_hat, sigma = model(
                batch['patient_idx'], batch['Z_prev'], batch['delta_t_prev'], 
                batch['has_prev'], batch['drug_indices'], batch['time_deltas'], batch['drug_mask']
            )
            
            # Negative Log-Likelihood Loss
            dist = torch.distributions.Normal(Z_hat, sigma)
            loss = -dist.log_prob(batch['Z_target']).sum(dim=1).mean()
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch['sample_idx'])
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                Z_hat, sigma = model(
                    batch['patient_idx'], batch['Z_prev'], batch['delta_t_prev'], 
                    batch['has_prev'], batch['drug_indices'], batch['time_deltas'], batch['drug_mask']
                )
                
                dist = torch.distributions.Normal(Z_hat, sigma)
                loss = -dist.log_prob(batch['Z_target']).sum(dim=1).mean()
                val_loss += loss.item() * len(batch['sample_idx'])
                
        if epoch % 100 == 0:
            tau_days = torch.exp(model.log_tau).item()
            mean_sigma = torch.exp(model.log_sigma).mean().item()
            print(f"Epoch {epoch:04d} | Train NLL: {train_loss/len(train_loader.dataset):.4f} | Val NLL: {val_loss/len(val_loader.dataset):.4f} | Tau: {tau_days:.1f}d | Mean Sigma: {mean_sigma:.3f}")
            
    torch.save(model.state_dict(), cache_dynamics)
    print(f"Saved Phase 2 model weights to {cache_dynamics}")
    return model

def evaluate_and_plot(dynamics_model, Theta, val_loader):
    device = next(dynamics_model.parameters()).device
    Theta = Theta.to(device)

    print("\nGenerating final validation predictions in relative abundance space...")
    dynamics_model.eval()
    
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            Z_hat, L = dynamics_model(
                batch['patient_idx'], batch['Z_prev'], batch['delta_t_prev'], 
                batch['has_prev'], batch['drug_indices'], batch['time_deltas'], batch['drug_mask']
            )
            
            logits = torch.matmul(Z_hat, Theta)
            P_hat = torch.softmax(logits, dim=1)
            
            y_true.append(batch['P_target'].cpu().numpy())
            y_pred.append(P_hat.cpu().numpy())
            
    y_true = np.concatenate(y_true).flatten()
    y_pred = np.concatenate(y_pred).flatten()
    
    mask = y_true > 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    log_y_true = np.log(y_true_filtered)
    log_y_pred = np.log(y_pred_filtered)
    
    corr = np.corrcoef(log_y_true, log_y_pred)[0, 1]
    print(f"Final Validation Pearson Correlation (log scale): {corr:.4f}")
    
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_filtered, y_pred_filtered, alpha=0.1, s=15, color='royalblue')
    plt.plot([1e-5, 1], [1e-5, 1], 'r--', label='Perfect Prediction (y=x)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-5, 1.1)
    plt.ylim(1e-5, 1.1)
    plt.xlabel('True Relative Abundance', fontsize=12)
    plt.ylabel('Predicted Relative Abundance', fontsize=12)
    plt.title(f'Phase 1 + Phase 2 Validation Predictions\nPearson r (log scale): {corr:.3f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "validation_scatter.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {plot_path}")

def save_model_results(dynamics_model, Theta, patient_list, patient2idx, asv_names, drug2idx, K=10, save_dir='.'):
    os.makedirs(save_dir, exist_ok=True)
    
    mu_p = dynamics_model.mu_p.weight.detach().cpu()
    sigma = torch.exp(dynamics_model.log_sigma).detach().cpu()
    cov_matrix = torch.diag(sigma ** 2)
    tau = torch.exp(dynamics_model.log_tau).detach().cpu()
    V_m = dynamics_model.V_m.weight.detach().cpu()
    Theta_tensor = Theta.detach().cpu() if isinstance(Theta, torch.Tensor) else torch.tensor(Theta)
    
    # 1. Save complete PyTorch checkpoint dictionary
    bundle_path = os.path.join(save_dir, f'model_params_K{K}.pt')
    torch.save({
        'K': K,
        'mu_p': mu_p,
        'sigma': sigma,
        'cov_matrix': cov_matrix,
        'log_sigma': dynamics_model.log_sigma.detach().cpu(),
        'tau': tau,
        'log_tau': dynamics_model.log_tau.detach().cpu(),
        'V_m': V_m,
        'Theta': Theta_tensor,
        'patient_list': patient_list,
        'patient2idx': patient2idx,
        'asv_names': asv_names,
        'drug2idx': drug2idx,
        'state_dict': dynamics_model.state_dict(),
    }, bundle_path)
    print(f"\nSaved complete model parameters to {bundle_path}")
    
    # 2. Save patient mean embeddings DataFrame (CSV)
    df_mu = pd.DataFrame(
        mu_p.numpy(),
        index=patient_list,
        columns=[f'latent_{i}' for i in range(K)]
    )
    df_mu.index.name = 'PatientID'
    csv_mu_path = os.path.join(save_dir, f'patient_mean_embeddings_K{K}.csv')
    df_mu.to_csv(csv_mu_path)
    print(f"Saved patient mean embeddings to {csv_mu_path}")
    
    # 3. Save covariance matrix (Numpy)
    cov_path = os.path.join(save_dir, f'global_covariance_K{K}.npy')
    np.save(cov_path, cov_matrix.numpy())
    print(f"Saved global covariance matrix to {cov_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print("\n" + "="*50)
    print(f"🚀 PyTorch Hardware Accelerator: {device.type.upper()}")
    print("="*50 + "\n")

    data_dir = '../Data'
    trajectory_df = pd.read_csv(os.path.join(data_dir, 'trajectory_data.csv'))
    
    asv_df_raw = pd.read_csv(os.path.join(data_dir, 'tblcounts_asv_wide.csv'))
    asv_df = asv_df_raw.set_index('ASV').T 
    asv_df = asv_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    rel_abund_df = asv_df.div(asv_df.sum(axis=1), axis=0).fillna(0)
    mean_rel_abund = rel_abund_df.mean(axis=0)
    asv_df = asv_df.loc[:, mean_rel_abund >= 1e-3]
    print(f"Filtered ASV shape: {asv_df.shape}")
    
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
    print(f"Train: {len(train_df)} | Val: {len(val_df)}")
    
    K_latent = 20
    
    train_dataset = MicrobiomeDataset(train_df, asv_df, patient2idx, drug2idx, max_drugs=20)
    val_dataset = MicrobiomeDataset(val_df, asv_df, patient2idx, drug2idx, max_drugs=20)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
    
    Theta_fixed = run_phase_1(train_loader, val_loader, asv_df.shape[1], K=K_latent, epochs=1500)
    
    dynamics_model = run_phase_2(
        train_loader, val_loader, 
        len(patient2idx), len(drug2idx), 
        K=K_latent, epochs=1000
    )
    
    evaluate_and_plot(dynamics_model, Theta_fixed, val_loader)
    
    save_model_results(
        dynamics_model=dynamics_model,
        Theta=Theta_fixed,
        patient_list=patient_list,
        patient2idx=patient2idx,
        asv_names=asv_df.columns.tolist(),
        drug2idx=drug2idx,
        K=K_latent,
        save_dir='.'
    )