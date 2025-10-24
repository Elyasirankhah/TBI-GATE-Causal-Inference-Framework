# TBI GATE Causal Inference Framework

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Group Average Treatment Effects (GATE) Analysis for Traumatic Brain Injury Rehabilitation**

This repository contains the methodological validation code for our study on heterogeneous treatment effects of early rehabilitation in traumatic brain injury (TBI) patients. The code implements a comprehensive three phase causal inference framework combining propensity score methods, X-learner meta-learning, and LLM-based counterfactual validation.


---

## 🎯 Overview

**Research Question:** Does early rehabilitation (≤3, ≤7, ≤14 days) after TBI improve patient outcomes, and do effects vary by baseline risk?

**Key Innovation:** Integration of medical language model embeddings (768-dimensional representations from TBIMS-pretrained DistilGPT-2) with traditional propensity score methods to capture complex patient heterogeneity.

**Main Findings:**
- **Phase 1 (ATE):** Early rehabilitation within 3 days reduces 30-day readmission risk by 10.1% (IPTW ATE = -0.101, 95% CI: [-0.159, -0.042])
- **Phase 2 (GATE):** Treatment effects are heterogeneous highest-risk patients (Q4) benefit most (GATE = -0.412), while low-risk patients (Q1-Q2) show harmful effects (+0.087, +0.056)
- **Phase 3 (LLM Validation):** Counterfactual predictions from Meta-Llama-3.1-8B align with traditional estimators after isotonic calibration
- **IHDP Benchmark:** sqrt(PEHE) = 0.493 after calibration

---

## 🔧 Installation

### Requirements
- Python 3.8 or higher
- pip package manager

### Setup

```bash
# Clone the repository
git clone https://github.com/Elyasirankhah/TBI-GATE-Causal-Inference-Framework.git
cd tbi-gate-causal-inference

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

Core packages:
- `numpy>=1.21.0`
- `pandas>=1.3.0`
- `scikit-learn>=1.0.0`
- `xgboost>=1.5.0`
- `matplotlib>=3.4.0`

Deep learning (for Siamese baseline):
- `torch>=1.10.0`

See `requirements.txt` for complete list.



---

## 🧪 Methodology

### Data Sources

### 🧠 [TBIMS National Database (Restricted Access)](https://tbindsc.org/TBIMS-National-Database)
- **Source:** Traumatic Brain Injury Model Systems (TBIMS), managed by TBINDSC  
- **N:** 79,604 patients with moderate-to-severe TBI (1987–2025)  
- **Access:** Requires formal data request and approval from TBINDSC  
- **Why restricted:** Contains de-identified clinical records; institutional approval required for use
- 
#### TBI Analysis Cohort (Not Publicly Available)
- **Source:** All of Us Research Program
- **N:** 24,939 adults with TBI (ICD-10: S06.0–S06.9, 2010–2024)
- **Access:** Requires approved All of Us Researcher Workbench access
- **Why restricted:** NIH Data Use Agreement protects participant privacy

#### IHDP Validation Dataset (Public)
- **Source:** [AMLab Amsterdam CEVAE Repository](https://github.com/AMLab-Amsterdam/CEVAE)
- **N:** 747 patients with 25 features
- **Ground Truth:** Known individual treatment effects (ITE)
- **Purpose:** Methodological validation without data access barriers


### IHDP Validation Details

**Why IHDP?**
- Public dataset with **known ground truth** treatment effects
- Allows calculation of Precision in Estimation of Heterogeneous Effects (PEHE)
- Standard benchmark used in 50+ causal inference papers

**Metrics:**
- **PEHE** = `√[ (1/n) Σᵢ (τ̂ᵢ - τᵢ)² ]` where `τᵢ = μ₁(xᵢ) - μ₀(xᵢ)` is true ITE
- **Correlation** = `corr(τ̂, τ)` (directional accuracy)
- **Sign Agreement** = fraction where `sign(τ̂ᵢ) = sign(τᵢ)`
- **Variance Ratio** = `Var(τ̂) / Var(τ)` (spread calibration)

**Calibration Techniques:**
1. Isotonic regression on validation set
2. Variance rescaling: `τ̂_rescaled = (τ̂ - mean(τ̂)) × [SD(τ_val) / SD(τ̂_val)] + mean(τ_val)`

**Robustness:** Multi-seed evaluation (n=5 seeds) confirms stability


---

## 📁 Repository Structure
```
tbi-gate-causal-inference/
├── README.md # This file
├── LICENSE # MIT License
├── requirements.txt # Python dependencies
│
├── TBI_Validation_Complete_Aligned.py 
├── ihdp_data.csv # IHDP benchmark data
```

---


### What's in This Repository (Public)

✅ **Methodological validation code** using public IHDP benchmark data
✅ **Complete X-learner implementation** with production parameters
✅ **Visualization tools** for GATE analysis
✅ **Documentation** of all methods and formulas

### What's NOT in This Repository

❌ **TBI patient analysis code** – uses All of Us Research Program data  
❌ **Raw patient data** – protected by NIH Data Use Agreement  
❌ **Medical embeddings** – requires TBIMS data access  
❌ **TBIMS National Database** – access only through [TBINDSC request portal](https://tbindsc.org/TBIMS-National-Database)


### How to Access TBI Analysis Code

**For Approved Researchers:**

If you have (or obtain) All of Us Researcher Workbench access:

1. Apply for access: [joinallofus.org/research](https://www.joinallofus.org/research)
2. Complete required training (CITI, data security)
3. Obtain institutional approval
4. Contact corresponding author for workspace sharing


**Why restricted?** NIH All of Us Data Use Agreement prohibits public release of code that accesses participant-level data. This is standard practice for all ~500 published All of Us studies.

---

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{irankhah2025threephase,
  title = {A Three-Phase Causal Inference Framework for Traumatic Brain Injury Rehabilitation: Integrating Medical LLM Embeddings, Heterogeneous Treatment Effects, and Counterfactual Validation},
  author = {Irankhah, Elyas and Pagare, Madhavi and Zhu, Yidong and Shen, Jiabin and Alam, Mohammad Arif Ul and Wolkowicz, Kelilah L.},
  journal = {Scientific Reports – AI for Clinical Decision-Making Collection},
  year = {2025},
}


---

## 🧩 Extending This Work

### Use Your Own Data


Replace IHDP data with your own observational study:

```python
# In gate_ite_validation_upgraded.py, replace load_ihdp_canonical() with:

def load_your_data():
    data = pd.read_csv("your_data.csv")
    
    # Required columns:
    # - treatment: binary (0/1)
    # - y_factual: observed outcome
    # - features: x1, x2, ..., xp
    
    # Optional (for validation with ground truth):
    # - y_cfactual: counterfactual outcome
    # - mu0: E[Y|T=0,X]
    # - mu1: E[Y|T=1,X]
    
    return data
```

### Adapt to Different Outcomes

**Continuous outcomes** (e.g., hospital length of stay):
- Remove `_clip01()` clipping
- Use `LinearRegression` or `GradientBoostingRegressor` for outcome models
- Metrics: MSE, R², predicted vs. actual plots

**Time-to-event outcomes** (e.g., survival):
- Use Cox proportional hazards models
- CATE = log hazard ratio
- Metrics: C-index, calibration slope

---

## 🐛 Troubleshooting

### Common Issues

**1. IHDP data download fails:**
```bash
# Manually download and place in working directory
wget https://raw.githubusercontent.com/AMLab-Amsterdam/CEVAE/master/datasets/IHDP/csv/ihdp_npci_1.csv
```

**2. ImportError for torch:**
```bash
# Siamese network requires PyTorch
pip install torch torchvision
```


---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📧 Contact

**Corresponding Author:** [Elyas Irankhah]
- Email: Elyas_irankhah@student.uml.edu
- ORCID: [0000-0001-7168-3898]
- 
💬 **Note:** If you would like access to the main analysis code or datasets based on the *All of Us Research Program* or *TBIMS National Database*, please contact the corresponding author directly.

**Project Link:** [https://github.com/Elyasirankhah/TBI-GATE-Causal-Inference-Framework]

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.







