# Time Series Anomaly Detection for IoT Sensors

This project focuses on detecting unusual patterns in IoT sensor data that could indicate equipment faults or maintenance needs.  
It combines both **statistical** and **deep learning** approaches — using **Isolation Forest** and an **LSTM Autoencoder** — to identify abnormal readings in multivariate time series data.

---

## Project Overview

Modern manufacturing systems rely on IoT sensors that continuously record data such as temperature, vibration, or pressure. Detecting anomalies in this data early helps prevent costly failures and supports predictive maintenance.

Since real sensor data with labeled anomalies is not always available, this project uses **synthetic time series data** that simulates real-world sensor behavior. The dataset includes:
- Normal periodic and trend patterns  
- Random noise  
- Injected anomalies (spikes, shifts, and dropouts)

The workflow includes:
1. Data generation and cleaning  
2. Exploratory data analysis (EDA)  
3. Feature engineering and scaling  
4. Model training (Isolation Forest and LSTM Autoencoder)  
5. Evaluation and visualization  


## Features

- Synthetic IoT sensor data generation  
- Automated anomaly injection  
- Rolling and lag-based feature engineering  
- Two anomaly detection models:
  - **Isolation Forest (unsupervised)**  
  - **LSTM Autoencoder (sequence reconstruction)**  
- Evaluation using precision, recall, and F1-score  
- Visual anomaly plots for both models  


## Models Used

### Isolation Forest
A tree-based unsupervised model that isolates abnormal points by randomly partitioning the data.  
It’s simple, fast, and effective for detecting outliers without labeled data.

### LSTM Autoencoder
A deep learning model that learns to reconstruct normal time sequences.  
Anomalies are detected when reconstruction error exceeds a defined threshold.  
The model captures temporal dependencies and performs better for sequence-based data.


## Results & Visualizations

Four key visualizations are automatically saved in the `plots/` folder:
1. `eda_preview.png` – sample of sensor readings  
2. `eda_corr.png` – correlation heatmap between sensors  
3. `isolation_forest.png` – anomalies detected by Isolation Forest  
4. `lstm_autoencoder.png` – anomalies detected by LSTM Autoencoder  

The results (precision, recall, F1-score, and thresholds) are stored in:
odels/out.json


## Folder Structure

anomaly_detect/
│
├── main.py
├── README.md
├── requirements.txt
├── Time_Series_Anomaly_Detection_Report.docx
│
├── models/
│   └── out.json
│
├── plots/
│   ├── eda_preview.png
│   ├── eda_corr.png
│   ├── isolation_forest.png
│   └── lstm_autoencoder.png
│
└── .gitignore


## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/dayanandahp/Time-Series-Anomaly-Detection-for-loT-Sensors.git

cd Folder name

2. Install dependencies
pip install -r requirements.txt

3. Run the project
python main.py

This will:
Generate the synthetic dataset
Train both models
Save evaluation metrics and plots



Requirements

Python 3.9+, 
Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn, tensorflow, statsmodels

(You can also install all dependencies using requirements.txt.)

Summary
This project demonstrates a complete anomaly detection pipeline suitable for predictive maintenance systems.
The Isolation Forest provides fast, interpretable detection, while the LSTM Autoencoder offers a more accurate, sequence-based approach for complex time series data.
Both models complement each other and can be easily extended to real IoT sensor datasets like NASA Bearing or AWS Server Metrics.

Project Report
A detailed project summary and explanation are included in:
Time_Series_Anomaly_Detection_Report.docx


GitHub Repository: https://github.com/dayanandahp/Time-Series-Anomaly-Detection-for-loT-Sensors


Project README: https://github.com/dayanandahp/Time-Series-Anomaly-Detection-for-loT-Sensors/blob/main/README.md

Author
Dayananda H P
B.E in Computer Science — SDMIT Ujire
Passionate about AI, Machine Learning, and Real-world Problem Solving
Email: dayanandahp06@gmail.com
