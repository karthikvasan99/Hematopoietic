import os
import json
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam

class MicrobiomeTrajectoryDataset(Dataset):
    def __init__(self, trajectory_df, asv_df, patient2idx, drug2idx, max_drugs=20):
        self.trajectory_df = trajectory_df.reset_index(drop=True)
        self.asv_df = asv_df
        self.patient2idx = patient2idx
        self.drug2idx = drug2idx
        self.max_drugs = max_drugs
        
        # Pre-compute target counts to save time during training
        self.target_counts = []
        for sample_id in self.trajectory_df['SampleID_Y']:
            counts = self.asv_df.loc[sample_id].values.astype(np.float32)
            self.target_counts.append(counts)

    def __len__(self):
        return len(self.trajectory_df)

    def __getitem__(self, idx):
        row = self.trajectory_df.iloc[idx]
        
        patient_idx = self.patient2idx[row['PatientID']]
        day_y = float(row['Day_Y'])
        
        # Parse historical drugs
        prior_drugs = json.loads(row['Prior_Drugs_30d'])
        
        drug_indices = []
        time_deltas = []
        drug_mask = []
        
        # Extract drug indices and time elapsed since administration
        for drug in prior_drugs:
            if len(drug_indices) >= self.max_drugs:
                break 
                
            drug_name = drug['Factor']
            start_day = float(drug['StartDayRelativeToNearestHCT'])
            
            drug_indices.append(self.drug2idx[drug_name])
            time_deltas.append(max(0.0, day_y - start_day))
            drug_mask.append(1.0)
            
        # Pad vectors
        while len(drug_indices) < self.max_drugs:
            drug_indices.append(0)
            time_deltas.append(0.0)
            drug_mask.append(0.0) 
            
        return {
            'patient_idx': torch.tensor(patient_idx, dtype=torch.long),
            'drug_indices': torch.tensor(drug_indices, dtype=torch.long),
            'time_deltas': torch.tensor(time_deltas, dtype=torch.float32),
            'drug_mask': torch.tensor(drug_mask, dtype=torch.float32),
            'counts': torch.tensor(self.target_counts[idx], dtype=torch.float32)
        }

def model(batch, num_patients, num_drugs, num_asvs, K=2):
    r"""
    Generative model mapping absolute drug histories to current microbiome states.
    Math aligns with: \tilde{z}_{pk} = \mu_{pk} + \sum_{m, t} V_{mk} e^{-t/\tau}
    """
    theta = pyro.sample("theta", dist.Normal(0., 1.).expand([K, num_asvs]).to_event(2))
    V = pyro.sample("V", dist.Normal(0., 1.).expand([num_drugs, K]).to_event(2))
    tau = pyro.sample("tau", dist.LogNormal(1., 1.)) 
    mu_p = pyro.sample("mu_p", dist.Normal(0., 1.).expand([num_patients, K]).to_event(2))

    patient_idx = batch['patient_idx']
    drug_idx = batch['drug_indices']
    t_deltas = batch['time_deltas']
    mask = batch['drug_mask']
    counts = batch['counts']

    mu_batch = mu_p[patient_idx]

    V_batch = V[drug_idx] 
    decay = torch.exp(-t_deltas / tau) 
    
    pert_batch = V_batch * decay.unsqueeze(-1) * mask.unsqueeze(-1)
    total_pert = pert_batch.sum(dim=1) 
    
    # Deterministic latent state
    mu_perturbed = mu_batch + total_pert

    with pyro.plate("data", counts.shape[0]):
        logits = torch.matmul(mu_perturbed, theta)
        p = torch.softmax(logits, dim=-1)
        
        total_counts = counts.sum(dim=-1, keepdim=True)
        expected_counts = total_counts * p
        
        pyro.sample("obs", dist.Poisson(expected_counts).to_event(1), obs=counts)
        
    # Return probabilities so we can extract them during evaluation
    return p

def train_model(train_df, val_df, asv_df, patient2idx, drug2idx, epochs=500, batch_size=32, K=2):
    num_patients = len(patient2idx)
    num_drugs = len(drug2idx)
    num_asvs = asv_df.shape[1]
    
    train_dataset = MicrobiomeTrajectoryDataset(train_df, asv_df, patient2idx, drug2idx)
    val_dataset = MicrobiomeTrajectoryDataset(val_df, asv_df, patient2idx, drug2idx)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    pyro.clear_param_store()
    guide = AutoDiagonalNormal(model)
    optimizer = Adam({"lr": 0.01})
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
    
    print("Starting training...")
    for epoch in range(epochs):
        epoch_train_loss = 0.0
        for batch in train_dataloader:
            loss = svi.step(batch, num_patients, num_drugs, num_asvs, K)
            epoch_train_loss += loss
        epoch_train_loss /= len(train_dataset)
        
        epoch_val_loss = 0.0
        for val_batch in val_dataloader:
            val_loss = svi.evaluate_loss(val_batch, num_patients, num_drugs, num_asvs, K)
            epoch_val_loss += val_loss
        epoch_val_loss /= len(val_dataset)
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | Train ELBO: {epoch_train_loss:.4f} | Val ELBO: {epoch_val_loss:.4f}")
            
    return model, guide

def evaluate_and_plot(trained_model, trained_guide, val_dataloader, num_patients, num_drugs, num_asvs, K):
    print("\nGenerating predictions for validation set...")
    
    # Predictive automatically samples the learned parameters from the guide and runs the model
    predictive = Predictive(trained_model, guide=trained_guide, num_samples=1, return_sites=("p",))
    
    true_abundances = []
    pred_abundances = []
    
    for batch in val_dataloader:
        # 1. Get true relative abundances
        counts = batch['counts']
        totals = counts.sum(dim=-1, keepdim=True)
        # Avoid division by zero
        true_p = counts / totals.clamp(min=1e-5) 
        
        # 2. Get predicted relative abundances
        preds = predictive(batch, num_patients, num_drugs, num_asvs, K)
        # preds['p'] has shape [num_samples, batch_size, num_asvs]. Squeeze out num_samples.
        pred_p = preds['p'].squeeze(0)
        
        true_abundances.append(true_p.detach().numpy())
        pred_abundances.append(pred_p.detach().numpy())
        
    # Flatten the arrays to plot every ASV in every sample
    y_true = np.concatenate(true_abundances).flatten()
    y_pred = np.concatenate(pred_abundances).flatten()
    
    # Filter out absolute zeros from the ground truth for cleaner log-scale visualization
    mask = y_true > 0
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]
    
    # Calculate Pearson correlation
    corr = np.corrcoef(y_true_filtered, y_pred_filtered)[0, 1]
    print(f"Validation Pearson Correlation (on non-zero true ASVs): {corr:.4f}")
    
    # Plotting
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_filtered, y_pred_filtered, alpha=0.1, s=15, color='royalblue')
    plt.plot([1e-5, 1], [1e-5, 1], 'r--', label='Perfect Prediction (y=x)')
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-5, 1.1)
    plt.ylim(1e-5, 1.1)
    
    plt.xlabel('True Relative Abundance (Log Scale)', fontsize=12)
    plt.ylabel('Predicted Relative Abundance (Log Scale)', fontsize=12)
    plt.title(f'Microbiome Composition Validation Predictions\nPearson r: {corr:.3f}', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = "validation_scatter.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Scatter plot saved to {plot_path}")

if __name__ == "__main__":
    data_dir = '../Data'
    trajectory_name = 'trajectory_data.csv'
    trajectory_path = os.path.join(data_dir, trajectory_name)
    trajectory_df = pd.read_csv(trajectory_path)
    
    asv_name = 'tblcounts_asv_wide.csv'
    asv_path = os.path.join(data_dir, asv_name)
    asv_df_raw = pd.read_csv(asv_path)
    
    print(f"Original ASV shape: {asv_df_raw.shape}")
    
    # Transpose so Index = SampleID, Columns = ASVs
    asv_df = asv_df_raw.set_index('ASV').T 
    asv_df = asv_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f"Transposed ASV shape: {asv_df.shape}")
    
    # --- ASV FILTERING ---
    print("Filtering ASVs by mean relative abundance >= 1e-3...")
    rel_abund_df = asv_df.div(asv_df.sum(axis=1), axis=0).fillna(0)
    mean_rel_abund = rel_abund_df.mean(axis=0)
    asv_df = asv_df.loc[:, mean_rel_abund >= 1e-3]
    print(f"Filtered ASV shape: {asv_df.shape}")
    # ---------------------
    
    valid_samples = set(asv_df.index)
    trajectory_df = trajectory_df[trajectory_df['SampleID_Y'].isin(valid_samples)]
    print(f"Ready to train on {len(trajectory_df)} samples and {asv_df.shape[1]} ASVs.")
    
    print("Building global patient and drug vocabularies...")
    patient_list = trajectory_df['PatientID'].unique().tolist()
    patient2idx = {pid: idx for idx, pid in enumerate(patient_list)}
    
    unique_drugs = set()
    for history_str in trajectory_df['Prior_Drugs_30d']:
        history = json.loads(history_str)
        for drug in history:
            unique_drugs.add(drug['Factor'])
            
    drug2idx = {drug: idx for idx, drug in enumerate(sorted(list(unique_drugs)))}
    
    print("Creating temporal train/validation splits...")
    train_dfs, val_dfs = [], []
    for pid, group in trajectory_df.groupby('PatientID'):
        group = group.sort_values('Day_Y')
        n_samples = len(group)
        if n_samples < 2:
            train_dfs.append(group)
            continue
        split_idx = int(n_samples * 0.8)
        if split_idx == n_samples:
            split_idx = n_samples - 1
        train_dfs.append(group.iloc[:split_idx])
        val_dfs.append(group.iloc[split_idx:])
        
    train_df = pd.concat(train_dfs)
    val_df = pd.concat(val_dfs)
    print(f"Train samples: {len(train_df)} | Val samples: {len(val_df)}")
    
    K_latent = 5
    trained_model, trained_guide = train_model(
        train_df, val_df, asv_df, patient2idx, drug2idx, epochs=500, batch_size=64, K=K_latent
    )
    print("Training complete!")
    
    # --- EVALUATION AND PLOTTING ---
    val_dataset = MicrobiomeTrajectoryDataset(val_df, asv_df, patient2idx, drug2idx)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    
    evaluate_and_plot(
        trained_model, 
        trained_guide, 
        val_dataloader, 
        len(patient2idx), 
        len(drug2idx), 
        asv_df.shape[1], 
        K=K_latent
    )