# 🌾 CropCast – Hybrid Crop Price Forecasting System

> AI-powered crop price forecasting using a Hybrid ARIMAX + Multi-Layer Perceptron (MLP) model for agricultural market intelligence.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Statsmodels](https://img.shields.io/badge/ARIMAX-Time%20Series-green)
![Supabase](https://img.shields.io/badge/Supabase-PostgreSQL-3ECF8E)
![License](https://img.shields.io/badge/License-MIT-blue)

---

# 📖 Overview

CropCast is an AI-driven crop price forecasting platform designed to help farmers, traders, researchers, and policymakers estimate future mandi prices before selling agricultural produce.

Unlike traditional forecasting systems that rely on a single statistical model, CropCast combines **ARIMAX** (for capturing linear trends) with a **Multi-Layer Perceptron Neural Network** (for learning nonlinear residual patterns), resulting in significantly improved forecasting performance for highly volatile crops.

The system currently supports:

- 🌾 Wheat
- 🍅 Tomato
- 🧅 Onion

across

- Gujarat
- Maharashtra

using historical AGMARKNET market data, engineered weather features, seasonal indicators, and market statistics.

---

# 🚨 Problem Statement

Agricultural commodity prices fluctuate dramatically due to:

- Seasonal production
- Rainfall variability
- Market arrivals
- Demand fluctuations
- Government policies

For example:

- Tomato prices can vary from ₹200 to ₹8,000 per quintal.
- Onion prices frequently collapse during harvest season.
- Farmers often sell without knowing future price trends.

CropCast aims to reduce this uncertainty through data-driven forecasting.

---

# ✨ Key Features

## Machine Learning

- Hybrid ARIMAX + MLP architecture
- Time-series forecasting
- Residual learning
- Feature engineering pipeline
- Chronological train-test split
- No data leakage

---

## Data Engineering

- Automated preprocessing pipeline
- Outlier removal using IQR
- Missing value handling
- Feature normalization
- Rolling statistics
- Lag features
- Seasonal encoding
- Weather integration

---

## Cloud Integration

- Supabase PostgreSQL database
- REST API uploads
- Prediction storage
- Metrics storage
- Feature storage

---

## Dashboard

Interactive dashboard displaying

- Crop prices
- Historical trends
- Forecasts
- Model comparison
- Performance metrics

---

# 🏗 System Architecture

```
                AGMARKNET Data
                      │
                      ▼
            Data Cleaning Pipeline
                      │
                      ▼
            Feature Engineering
                      │
      ┌───────────────┴───────────────┐
      ▼                               ▼
  ARIMAX Model                  Weather Features
      │
      ▼
Residual Errors
      │
      ▼
MLP Neural Network
      │
      ▼
Hybrid Prediction
      │
      ▼
Supabase PostgreSQL
      │
      ▼
Interactive Dashboard
```

---

# ⚙ Tech Stack

## Languages

- Python
- SQL

## Machine Learning

- Scikit-Learn
- Statsmodels
- NumPy
- Pandas

## Database

- Supabase PostgreSQL

## Visualization

- Matplotlib
- HTML
- CSS
- JavaScript

## Deployment

- Supabase
- GitHub

---

# 📊 Dataset

Source:

**AGMARKNET**

Contains historical mandi prices from Gujarat and Maharashtra.

### Crops

- Wheat
- Tomato
- Onion

### States

- Gujarat
- Maharashtra

### Total Raw Records

55,392+

---

# 🔬 Feature Engineering

The model generates 26 engineered features including:

### Lag Features

- 1 day
- 3 day
- 7 day
- 14 day
- 30 day

### Rolling Statistics

- Mean
- Standard Deviation
- Maximum
- Minimum

### Weather

- Rainfall
- Temperature
- Climate routing

### Seasonality

- Month
- Quarter
- Week of Year
- Sin/Cos encoding

### Market Indicators

- Price Momentum
- Price Spread
- Log Arrivals

---

# 🧠 Hybrid Model

CropCast combines two models.

## Stage 1

ARIMAX predicts the linear component of crop prices.

```
Actual Price
      │
      ▼
   ARIMAX
      │
      ▼
Prediction
```

Residual error:

```
Residual = Actual − Prediction
```

---

## Stage 2

The residuals are passed into a Multi-Layer Perceptron.

```
Residuals
      │
      ▼
MLP Neural Network
      │
      ▼
Residual Prediction
```

---

## Final Prediction

```
Hybrid Prediction

=

ARIMAX Prediction

+

MLP Residual Correction
```

This allows

- ARIMAX to learn linear relationships
- MLP to capture nonlinear patterns

---

# 📈 Results

| Crop | Best Model | Accuracy |
|-------|------------|-----------|
| Wheat | ARIMAX | 97.88% |
| Onion | Hybrid | 86.07% |
| Tomato | Hybrid | 79.91% |

Hybrid model reduced RMSE by

- Tomato → **43.9%**
- Onion → **38.9%**

compared to standalone ARIMAX.

---

# 🏆 Project Highlights

✔ Hybrid ARIMAX + Neural Network

✔ Weather-aware forecasting

✔ Multi-state forecasting

✔ Residual learning

✔ Automated feature engineering

✔ Cloud database integration

✔ Interactive dashboard

✔ Time-series validation

✔ Production-ready pipeline

---

# 📂 Project Structure

```
CropCast/

│
├── data/
├── notebooks/
├── models/
├── preprocessing/
├── forecasting/
├── dashboard/
├── database/
├── outputs/
├── images/
├── requirements.txt
└── README.md
```

---

# 📊 Outputs

The pipeline generates:

- Predictions CSV
- Model Metrics
- Feature Importance
- Actual vs Predicted plots
- Model Comparison charts
- Residual Analysis

---

# 🚀 Future Improvements

- Real-time AGMARKNET API integration
- Daily automatic retraining
- Mobile application
- Farmer recommendation engine
- Demand forecasting
- Crop profitability prediction
- Fertilizer recommendation
- AI chatbot for farmers
- Multi-language support

---

# 👨‍💻 Author

**V.Aditya Yaswardhan**

B.Tech Computer Science

SRM Institute of Science and Technology

Interested in

- Artificial Intelligence
- Machine Learning
- Data Engineering
- Generative AI

---

# ⭐ If you found this project useful

Please consider giving it a ⭐ on GitHub.
