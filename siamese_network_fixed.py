# ============================================================================
# Siamese Network Validation - FIXED VERSION
# Corrected implementation for fair comparison with GATE methodology
# ============================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.isotonic import IsotonicRegression
import warnings
warnings.filterwarnings('ignore')

print("=== Siamese Network Validation - FIXED VERSION ===")
print("Corrected implementation for fair comparison with GATE")
print()

# ============================================================================
# Fixed Siamese Network Architecture
# ============================================================================
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_prob=0.5):
        super().__init__()
        # Shared representation
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        # Two heads off shared representation
        self.head0 = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(100, 1)
        )
        self.head1 = nn.Sequential(
            nn.Linear(hidden_dim, 100),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        h = self.shared(x)
        return self.head0(h), self.head1(h)  # y0_hat, y1_hat

# ============================================================================
# Fixed Training and Evaluation
# ============================================================================
def train_siamese_network_fixed(data, features, treatment_col, outcome_col, 
                               hidden_dim=200, dropout_prob=0.5, epochs=100, 
                               learning_rate=0.001, device='cpu'):
    """Train Siamese network with correct tensor handling"""
    
    # Prepare data
    X = data[features].values
    t = data[treatment_col].values
    y = data[outcome_col].values
    
    # Separate treatment and control groups
    t0_data = data[data[treatment_col] == 0].copy()
    t1_data = data[data[treatment_col] == 1].copy()
    
    print(f"Training data: {len(t0_data)} control, {len(t1_data)} treated")
    
    # Convert to tensors - FIXED: each group gets its own features
    X_t0 = torch.FloatTensor(t0_data[features].values)
    X_t1 = torch.FloatTensor(t1_data[features].values)
    y_t0 = torch.FloatTensor(t0_data[outcome_col].values).unsqueeze(1)
    y_t1 = torch.FloatTensor(t1_data[outcome_col].values).unsqueeze(1)
    
    # Create model
    model = SiameseNetwork(len(features), hidden_dim, dropout_prob).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # Training loop - FIXED: each head trained on its own group
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # FIXED: Control head trained on control group
        y0_pred_t0, _ = model(X_t0)  # use the y0 head
        
        # FIXED: Treatment head trained on treatment group  
        _, y1_pred_t1 = model(X_t1)  # use the y1 head
        
        # FIXED: Loss computed on matching subgroups
        loss = criterion(y0_pred_t0, y_t0) + criterion(y1_pred_t1, y_t1)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model

def predict_treatment_effects_fixed(model, data, features, device='cpu'):
    """Predict treatment effects with correct counterfactual inputs"""
    
    X = torch.FloatTensor(data[features].values).to(device)
    
    model.eval()
    with torch.no_grad():
        # FIXED: Always use both heads on every x, then subtract
        y0_hat, y1_hat = model(X)
        ite = (y1_hat - y0_hat).cpu().numpy().ravel()
    
    return ite

def calculate_enhanced_metrics_fixed(ite_pred, ite_true):
    """Calculate enhanced metrics for fair comparison with GATE"""
    
    # Basic metrics
    pehe = float(np.mean((ite_pred - ite_true) ** 2))
    sqrt_pehe = float(np.sqrt(pehe))
    mse = float(np.mean((ite_pred - ite_true) ** 2))
    mae = float(np.mean(np.abs(ite_pred - ite_true)))
    correlation = float(np.corrcoef(ite_pred, ite_true)[0, 1]) if len(ite_pred) > 1 else 0.0
    sign_agreement = float(np.mean(np.sign(ite_pred) == np.sign(ite_true)))
    
    # Variance analysis
    sd_true = float(np.std(ite_true))
    sd_pred = float(np.std(ite_pred))
    variance_ratio = sd_pred / sd_true if sd_true > 0 else float('nan')
    
    print(f"PEHE: {pehe:.6f}")
    print(f"sqrt(PEHE): {sqrt_pehe:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Sign Agreement: {sign_agreement:.3f}")
    print(f"Variance Analysis:")
    print(f"  sd(tau_true): {sd_true:.4f}")
    print(f"  sd(tau_pred): {sd_pred:.4f}")
    print(f"  Ratio: {variance_ratio:.4f}")
    
    return {
        'pehe': pehe,
        'sqrt_pehe': sqrt_pehe,
        'mse': mse,
        'mae': mae,
        'correlation': correlation,
        'sign_agreement': sign_agreement,
        'sd_true': sd_true,
        'sd_pred': sd_pred,
        'variance_ratio': variance_ratio
    }

def learn_calibration_mappings_siamese(data_train, data_val, features, treatment_col, outcome_col):
    """Learn calibration mappings on validation set only (like GATE)"""
    print("=== Learning Calibration Mappings on Validation Set ===")
    
    # Train model on training data
    model = train_siamese_network_fixed(data_train, features, treatment_col, outcome_col)
    
    # Get predictions on validation set
    pred_effects_val = predict_treatment_effects_fixed(model, data_val, features)
    true_effects_val = (data_val['mu1'] - data_val['mu0']).values
    
    # Learn variance rescaling factor on validation set
    a = np.std(true_effects_val) / np.std(pred_effects_val) if np.std(pred_effects_val) > 0 else 1.0
    
    # Learn isotonic calibration on validation set
    iso = IsotonicRegression(out_of_bounds='clip')
    iso.fit(pred_effects_val, true_effects_val)
    
    print(f"Variance rescaling factor: {a:.4f}")
    print(f"Isotonic calibration learned on {len(pred_effects_val)} validation samples")
    
    return model, a, iso

def apply_calibration_mappings_siamese(pred_effects, a, iso):
    """Apply learned mappings to test set"""
    # Variance rescaling
    pred_rescaled = a * pred_effects
    
    # Isotonic calibration
    pred_calibrated = iso.predict(pred_effects)
    
    return pred_rescaled, pred_calibrated

# ============================================================================
# Fixed IHDP Validation with Proper Train/Val/Test Split
# ============================================================================
def validate_siamese_fixed():
    """Validate fixed Siamese network on IHDP with proper splits"""
    
    print("=== Fixed Siamese Network Validation on IHDP ===")
    
    # Load IHDP data
    data = pd.read_csv("https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv", header=None)
    
    # Set column names
    col = ["treatment", "y_factual", "y_cfactual", "mu0", "mu1"]
    for i in range(1, 26):
        col.append("x" + str(i))
    data.columns = col
    data = data.astype({"treatment": 'bool'}, copy=False)
    
    print(f"IHDP data shape: {data.shape}")
    print(f"True ATE: {(data['mu1'] - data['mu0']).mean():.4f}")
    
    # Proper 3-way split: 60/20/20 (train/val/test) - same as GATE
    np.random.seed(42)
    n = len(data)
    indices = np.random.permutation(n)
    
    train_size = int(0.6 * n)
    val_size = int(0.2 * n)
    
    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]
    
    data_train = data.iloc[train_idx].reset_index(drop=True)
    data_val = data.iloc[val_idx].reset_index(drop=True)
    data_test = data.iloc[test_idx].reset_index(drop=True)
    
    print(f"Train: {len(data_train)}, Val: {len(data_val)}, Test: {len(data_test)}")
    
    # Prepare features
    features = [f'x{i}' for i in range(1, 26)]
    
    # Learn calibration mappings on validation set
    print("\n1. LEARNING CALIBRATION MAPPINGS")
    print("-" * 50)
    model, a, iso = learn_calibration_mappings_siamese(data_train, data_val, features, 'treatment', 'y_factual')
    
    # Baseline evaluation on test set
    print("\n2. BASELINE EVALUATION (Test Set)")
    print("-" * 50)
    pred_effects_test = predict_treatment_effects_fixed(model, data_test, features)
    true_effects_test = (data_test['mu1'] - data_test['mu0']).values
    
    metrics_baseline = calculate_enhanced_metrics_fixed(pred_effects_test, true_effects_test)
    baseline_pehe = metrics_baseline['sqrt_pehe']
    
    # Isotonic calibration (learned on validation, applied to test)
    print("\n3. ISOTONIC CALIBRATION (Rigorous)")
    print("-" * 50)
    _, pred_calibrated = apply_calibration_mappings_siamese(pred_effects_test, a, iso)
    metrics_calibrated = calculate_enhanced_metrics_fixed(pred_calibrated, true_effects_test)
    calibrated_pehe = metrics_calibrated['sqrt_pehe']
    
    # Final comparison
    print("\n" + "="*70)
    print("FIXED SIAMESE NETWORK RESULTS")
    print("="*70)
    print(f"Baseline sqrt(PEHE): {baseline_pehe:.4f}")
    print(f"Isotonic sqrt(PEHE): {calibrated_pehe:.4f}")
    print(f"Improvement: {((baseline_pehe - calibrated_pehe) / baseline_pehe * 100):+.1f}%")
    
    # Performance assessment (internal threshold)
    if calibrated_pehe <= 0.6:
        print("EXCELLENT: Meets our internal threshold for excellent performance")
    elif calibrated_pehe <= 0.8:
        print("VERY GOOD: Strong performance")
    elif calibrated_pehe <= 1.0:
        print("GOOD: Solid performance")
    else:
        print("MODERATE: Room for improvement")
    
    print("\nComparison with GATE:")
    print(f"GATE sqrt(PEHE): 0.365 (state-of-the-art)")
    print(f"Siamese sqrt(PEHE): {calibrated_pehe:.3f}")
    
    if calibrated_pehe < 0.365:
        print("Siamese Network: BETTER than GATE!")
    else:
        improvement_factor = calibrated_pehe / 0.365
        print(f"GATE is {improvement_factor:.1f}x BETTER than Siamese Network")
    
    return {
        'baseline_pehe': baseline_pehe,
        'calibrated_pehe': calibrated_pehe,
        'correlation': metrics_calibrated['correlation'],
        'sign_agreement': metrics_calibrated['sign_agreement']
    }

# ============================================================================
# Main Execution
# ============================================================================
if __name__ == "__main__":
    print("Starting FIXED Siamese Network Validation...")
    print("="*70)
    
    # Run fixed validation
    results = validate_siamese_fixed()
    
    print("\n" + "="*70)
    print("FINAL COMPARISON SUMMARY")
    print("="*70)
    print("Methodology Comparison:")
    print(f"GATE (X-learner + Medical embeddings): sqrt(PEHE) = 0.365")
    print(f"Siamese Network (Fixed): sqrt(PEHE) = {results['calibrated_pehe']:.3f}")
    print()
    print("Both methods now use:")
    print("- Proper train/val/test splits")
    print("- Calibration learned on validation, applied to test")
    print("- Same evaluation metrics")
    print("- Rigorous validation methodology")
    print("="*70)
