import os
import json
import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
import itertools
import random

def compute_dataset_stats(data_dir=None):
    if data_dir is None:
        # Check relative and absolute paths
        if os.path.exists('../Data'):
            data_dir = '../Data'
        elif os.path.exists('./Data'):
            data_dir = './Data'
        elif os.path.exists('/home/karthik_unix/Hematopoietic/Data'):
            data_dir = '/home/karthik_unix/Hematopoietic/Data'
        else:
            raise FileNotFoundError("Could not locate Data directory.")
            
    print(f"Loading data from {data_dir}...")
    trajectory_df = pd.read_csv(os.path.join(data_dir, 'trajectory_data.csv'))
    asv_df_raw = pd.read_csv(os.path.join(data_dir, 'tblcounts_asv_wide.csv'))
    
    # Process ASV counts and filter as in model_fit.py
    asv_df = asv_df_raw.set_index('ASV').T 
    asv_df = asv_df.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    # Relative abundance
    rel_abund_df = asv_df.div(asv_df.sum(axis=1), axis=0).fillna(0)
    mean_rel_abund = rel_abund_df.mean(axis=0)
    
    # Filter ASVs with mean relative abundance >= 1e-3 (same as model_fit.py)
    filtered_asv_df = rel_abund_df.loc[:, mean_rel_abund >= 1e-3]
    
    # Normalize filtered relative abundances so each sample sums to 1
    P_df = filtered_asv_df.div(filtered_asv_df.sum(axis=1), axis=0).fillna(0)
    
    valid_samples = set(P_df.index)
    filtered_traj_df = trajectory_df[trajectory_df['SampleID_Y'].isin(valid_samples)].copy()
    
    print("\n" + "="*60)
    print("1. PATIENT & SAMPLE SUMMARY STATISTICS")
    print("="*60)
    
    # Raw trajectory dataset
    unique_patients_raw = trajectory_df['PatientID'].nunique()
    samples_per_patient_raw = trajectory_df.groupby('PatientID')['SampleID_Y'].nunique()
    
    print(f"Raw Trajectory Data:")
    print(f"  • Unique Patients: {unique_patients_raw}")
    print(f"  • Total Samples: {len(trajectory_df)}")
    print(f"  • Samples per Patient:")
    print(f"      - Mean:   {samples_per_patient_raw.mean():.2f}")
    print(f"      - Median: {samples_per_patient_raw.median():.2f}")
    print(f"      - Std:    {samples_per_patient_raw.std():.2f}")
    print(f"      - Min:    {samples_per_patient_raw.min()}")
    print(f"      - Max:    {samples_per_patient_raw.max()}")
    print(f"      - 25th %: {samples_per_patient_raw.quantile(0.25):.2f}")
    print(f"      - 75th %: {samples_per_patient_raw.quantile(0.75):.2f}")
    
    # Filtered model dataset
    unique_patients_filt = filtered_traj_df['PatientID'].nunique()
    samples_per_patient_filt = filtered_traj_df.groupby('PatientID')['SampleID_Y'].nunique()
    
    print(f"\nFiltered Model Data (Samples matching filtered ASV matrix):")
    print(f"  • Unique Patients: {unique_patients_filt}")
    print(f"  • Total Samples: {len(filtered_traj_df)}")
    print(f"  • Filtered ASV Dimensions: {filtered_asv_df.shape[1]}")
    print(f"  • Samples per Patient:")
    print(f"      - Mean:   {samples_per_patient_filt.mean():.2f}")
    print(f"      - Median: {samples_per_patient_filt.median():.2f}")
    print(f"      - Std:    {samples_per_patient_filt.std():.2f}")
    print(f"      - Min:    {samples_per_patient_filt.min()}")
    print(f"      - Max:    {samples_per_patient_filt.max()}")
    print(f"      - 25th %: {samples_per_patient_filt.quantile(0.25):.2f}")
    print(f"      - 75th %: {samples_per_patient_filt.quantile(0.75):.2f}")
    
    print("\n" + "="*60)
    print("2. JENSEN-SHANNON DIVERGENCE (JSD): INTRA VS. INTER-PATIENT")
    print("="*60)
    
    patient_to_samples = filtered_traj_df.groupby('PatientID')['SampleID_Y'].apply(list).to_dict()
    sample_profiles = {s_id: P_df.loc[s_id].values for s_id in filtered_traj_df['SampleID_Y'].unique() if s_id in P_df.index}
    
    # 2a. Intra-person JSD
    intra_jsd_list = []
    intra_js_dist_list = []
    
    for pid, s_ids in patient_to_samples.items():
        valid_s = [s for s in s_ids if s in sample_profiles]
        if len(valid_s) < 2:
            continue
        for s1, s2 in itertools.combinations(valid_s, 2):
            p1, p2 = sample_profiles[s1], sample_profiles[s2]
            js_dist = jensenshannon(p1, p2, base=2)
            js_div = js_dist ** 2
            intra_js_dist_list.append(js_dist)
            intra_jsd_list.append(js_div)
            
    # 2b. Inter-person / Random JSD
    inter_jsd_list = []
    inter_js_dist_list = []
    patient_ids = [pid for pid, s_ids in patient_to_samples.items() if len(s_ids) > 0]
    num_inter_samples = min(50000, len(intra_jsd_list) * 5 if len(intra_jsd_list) > 0 else 50000)
    
    random.seed(42)
    for _ in range(num_inter_samples):
        p_a, p_b = random.sample(patient_ids, 2)
        s1 = random.choice(patient_to_samples[p_a])
        s2 = random.choice(patient_to_samples[p_b])
        if s1 in sample_profiles and s2 in sample_profiles:
            p1, p2 = sample_profiles[s1], sample_profiles[s2]
            js_dist = jensenshannon(p1, p2, base=2)
            js_div = js_dist ** 2
            inter_js_dist_list.append(js_dist)
            inter_jsd_list.append(js_div)
            
    intra_jsd = np.array(intra_jsd_list)
    inter_jsd = np.array(inter_jsd_list)
    intra_dist = np.array(intra_js_dist_list)
    inter_dist = np.array(inter_js_dist_list)
    
    print(f"Pairs Analyzed: {len(intra_jsd):,} intra-patient pairs | {len(inter_jsd):,} inter-patient pairs")
    
    print("\n--- Jensen-Shannon Divergence (JSD, base 2, range [0, 1]) ---")
    print(f"  • Intra-patient JSD: Mean = {intra_jsd.mean():.4f} ± {intra_jsd.std():.4f} (Median = {np.median(intra_jsd):.4f})")
    print(f"  • Inter-patient JSD: Mean = {inter_jsd.mean():.4f} ± {inter_jsd.std():.4f} (Median = {np.median(inter_jsd):.4f})")
    print(f"  • Ratio (Inter / Intra): {inter_jsd.mean() / (intra_jsd.mean() + 1e-8):.2f}x")
    
    print("\n--- Jensen-Shannon Distance (sqrt(JSD)) ---")
    print(f"  • Intra-patient JS Distance: Mean = {intra_dist.mean():.4f} ± {intra_dist.std():.4f} (Median = {np.median(intra_dist):.4f})")
    print(f"  • Inter-patient JS Distance: Mean = {inter_dist.mean():.4f} ± {inter_dist.std():.4f} (Median = {np.median(inter_dist):.4f})")
    print(f"  • Ratio (Inter / Intra): {inter_dist.mean() / (intra_dist.mean() + 1e-8):.2f}x")
    print("="*60 + "\n")

if __name__ == '__main__':
    compute_dataset_stats()
