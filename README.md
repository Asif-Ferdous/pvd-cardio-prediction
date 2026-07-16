# Personalized Cardiovascular Risk Assessment
### A Dual-Stage Machine Learning Framework for Individualized PVD Impact Quantification

A machine-learning system that screens for **Peripheral Vascular Disease (PVD)** and then estimates **Cardiovascular Disease (CVD)** risk with PVD modeled as an explicit feature — and, crucially, quantifies *how much PVD raises each individual patient's CVD risk* through counterfactual inference. Built on the **CRISP-DM** methodology and deployed as a web-based clinical prototype.

> **Status:** Manuscript in preparation. Thesis title: *"Personalized Cardiovascular Risk Assessment: A Dual-Stage Machine Learning Framework for Individualized PVD Impact Quantification."*

**🔗 Live demo:** _add your Render app URL here_

---

## Why this project

Cardiovascular disease causes ~17.9 million deaths a year (≈31% of global deaths). Peripheral vascular disease is both an independent atherosclerotic condition and a strong contributor to cardiovascular events — yet **40–60% of PVD cases go undiagnosed** in primary care because gold-standard testing (e.g. ankle-brachial index) is resource-heavy.

Conventional risk calculators (Framingham, QRISK3, ASCVD, SCORE2) report **population-level** risk and can't answer the question a clinician actually cares about:

> *"How much does PVD increase **this particular patient's** cardiovascular risk?"*

This project answers that question with a two-stage, PVD-aware pipeline and patient-level counterfactual analysis.

## Approach at a glance

```
Patient input — 11 routine clinical & lifestyle features
        │
        ▼
┌────────────────────────────┐
│ Stage 1 — PVD screening    │   Predict PVD from low-cost primary-care features
└─────────────┬──────────────┘
              │  PVD status
              ▼
┌────────────────────────────┐
│ Stage 2 — CVD prediction   │   Predict CVD risk WITH PVD as an explicit feature
└─────────────┬──────────────┘
              │
              ▼
┌────────────────────────────────────────────────────────────┐
│ Counterfactual simulation (per patient):                   │
│   risk if PVD = 0   vs.   risk if PVD = 1                   │
│   → individual PVD-attributable increase in CVD risk        │
└────────────────────────────────────────────────────────────┘
```

The Stage-2 model is trained *with* PVD as a feature, so for any patient we can flip PVD off and on while holding every other variable constant, and read off the relative change in predicted CVD risk. This is the study's core contribution: moving from population averages to **individualized, PVD-aware risk quantification**.

## Methodology (CRISP-DM)

| Phase | Summary |
|-------|---------|
| **Business understanding** | Frame PVD screening + CVD risk as a linked clinical decision-support problem; target >90% recall for PVD screening and calibrated CVD probabilities. |
| **Data understanding** | Public cardiovascular dataset from Mendeley Data (Banerjee 2024; Siddhartha 2020), **70,002 patient records**, 11 baseline features. |
| **Data preparation** | Range/physiology validation, age-days→years conversion, BMI computation, MinMax scaling, SMOTE for class imbalance, and PVD-label derivation via a clinical scoring rubric. |
| **Modeling** | Six algorithms benchmarked per stage (see below). |
| **Evaluation** | Accuracy, Precision, Recall, F1, AUC-ROC — with recall prioritized for PVD screening and AUC-ROC/F1 for CVD. |
| **Deployment** | Flask + Bootstrap 5 web prototype with input validation and dynamic model loading. |

## Dataset

- **Source:** Mendeley Data (Banerjee, 2024; Siddhartha, 2020)
- **Records:** 70,002 (≈94–96% retained after cleaning)
- **Targets:** CVD (original label), PVD (derived)
- **PVD prevalence:** ~22% (from scoring rubric)
- **CVD prevalence:** ~49.5% overall; ~70.3% among PVD-positive vs ~40.3% among PVD-negative

**The 11 features:** age, gender, height, weight, systolic BP (`ap_hi`), diastolic BP (`ap_lo`), cholesterol, glucose, smoking, alcohol, physical activity.

### PVD labeling — a transparent scoring rubric

The dataset has no PVD label, so one is derived from established risk factors:

| Risk factor | Points | Rationale |
|-------------|:------:|-----------|
| Age > 50 | +1 | Age-dependent PVD risk |
| Smoking | +2 | Strongest modifiable risk factor |
| Hypertension | +1 | Accelerates atherosclerosis |
| Elevated cholesterol | +1 | Lipid-driven plaque formation |
| Obesity (BMI > 30) | +1 | Metabolic/inflammatory pathways |
| Physical inactivity | +1 | Reduced vascular health |

**PVD-positive when total score > 3**, yielding ~22% prevalence — consistent with population estimates for older adults with cardiovascular risk factors.

## Models

Decision Tree · Random Forest · Stochastic Gradient Descent (SGD) · XGBoost · Neural Network (MLP) · **TabNet** (attention-based deep learning for tabular data, providing interpretable feature-attention weights).

## Results

> Numbers below are transcribed from the thesis. A couple of decimals were read via OCR — **verify against `model_metrics.txt` / the performance CSVs before publishing.**

### Stage 1 — PVD screening

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| **TabNet** | **99.5%** | **99.6%** | **99.3%** | **99.4%** | **99.5%** |
| Neural Network (MLP) | 93.0% | 93.9% | 88.8% | 91.3% | 92.4% |
| Random Forest | 89.6% | 90.6% | 83.4% | 86.9% | 88.6% |
| SGD | 84.9% | 88.7% | 72.8% | 79.9% | 83.1% |
| Decision Tree | 82.6% | 83.5% | 72.2% | 77.4% | 81.1% |
| XGBoost | 77.6% | 93.4% | 49.2% | 64.5% | 73.4% |

**TabNet leads decisively** — 99.3% recall means only ~0.7% of true PVD cases are missed, which is exactly what a screening tool needs. XGBoost, despite high precision, was too conservative (49.2% recall) under its hyperparameter configuration.

### Stage 2 — CVD prediction (with PVD feature)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|
| **XGBoost** | **73.3%** | **75.5%** | **69.2%** | **72.2%** | **73.3%** |
| Neural Network (MLP) | 73.2% | 74.3% | 71.4% | 72.7% | 73.2% |
| TabNet | 72.9% | 73.4% | 72.2% | 72.8% | 72.9% |
| Decision Tree | 72.7% | 75.5% | 67.5% | 71.3% | 72.7% |
| SGD | 72.7% | 75.7% | 67.1% | 71.4% | 72.7% |
| Random Forest | 72.6% | 75.3% | 67.5% | 71.2% | 72.6% |

All six models cluster tightly (72.6–73.3%), suggesting the feature set captures a stable CVD signal with diminishing returns from added complexity. Models that include PVD outperform those without it, confirming PVD carries real predictive value.

### Individualized PVD impact (counterfactual analysis)

For each patient we compute CVD risk under PVD-absent vs PVD-present scenarios and report the relative increase:

```
Relative Risk Increase (%) = [ P(CVD | x, PVD=1) − P(CVD | x, PVD=0) ]  /  P(CVD | x, PVD=0) × 100
```

Observed patient-level increases range from **~10% to >150%**, with a **cohort average of ~74%** — heterogeneity that population-level calculators completely hide.

## Tech stack

- **Language:** Python
- **ML / DL:** scikit-learn, XGBoost, PyTorch-TabNet, MLP
- **Imbalance handling:** SMOTE
- **Web app:** Flask, Bootstrap 5, HTML templates
- **Deployment:** Render (`render.yaml`, `Procfile`)

## Project structure

```
pvd-cardio-prediction/
├── app.py                              # Flask web application
├── predict_pvd.py                      # PVD inference logic
├── training_script.py                  # Model training
├── data_mining.py                      # Data prep / feature engineering
├── models/                             # Saved trained models
├── templates/                          # Web app HTML (Bootstrap 5)
├── converted_cardio_train_with_PVD_years.csv
├── feature_names_cardio.csv
├── feature_names_pvd.csv
├── model_metrics.txt
├── model_performance_metrics_cardio.csv
├── model_performance_metrics_pvd.csv
├── render.yaml / Procfile              # Deployment config
└── requirements.txt
```

## Run it locally

```bash
git clone https://github.com/Asif-Ferdous/pvd-cardio-prediction.git
cd pvd-cardio-prediction
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open the local URL printed by Flask and enter patient values to get a PVD-aware CVD risk assessment.

## Limitations & future work

**Data**
- Single-dataset validation; external validation on diverse cohorts is needed.
- Cross-sectional design limits temporal risk-trajectory inference.
- Derived PVD labels lack gold-standard confirmation (ABI, imaging).
- Feature set limited to 11 basic variables.

**Methodology**
- SMOTE synthetic samples may introduce artifacts.
- Conservative, lightly searched hyperparameters may underfit.
- Counterfactual validity assumes correct model extrapolation.
- Calibration assessment is needed beyond discrimination metrics.

**Future directions**
- External validation and longitudinal extension with real cardiovascular events.
- Validate the PVD rubric against ABI-confirmed diagnoses.
- Feature augmentation (biomarkers, imaging, genetics).
- Fairness auditing across protected groups; causal-inference methods for more robust impact estimation.

## Author

**M M Asif Ferdous** — www.linkedin.com/in/m-m-asif-ferdous-108774b6 · [GitHub](https://github.com/Asif-Ferdous)

## Citation

```
Ferdous, M. M. A. (2026). Personalized Cardiovascular Risk Assessment:
A Dual-Stage Machine Learning Framework for Individualized PVD Impact
Quantification. Manuscript in preparation.
```
