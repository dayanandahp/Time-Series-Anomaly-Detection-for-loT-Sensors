

import os, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from statsmodels.tsa.seasonal import seasonal_decompose


np.random.seed(42)
os.makedirs('models', exist_ok=True)            # Setup
os.makedirs('plots', exist_ok=True)

def gen_data(n=20000, s=3, ar=0.01):
    
    t = pd.date_range('2024-01-01', periods=n, freq='T')         # Create timestamp 
    df = pd.DataFrame({'time': t})


    f = np.linspace(0.0005, 0.005, s)
    for i in range(s):
        tr = 0.00005 * np.arange(n)                     # slow trend
        se = 2 * np.sin(np.arange(n) * 2 * np.pi * f[i]) # periodic pattern
        ns = 0.5 * np.random.randn(n)                   # random noise
        df[f's{i+1}'] = 10 + tr + se + ns

   
    lbl = np.zeros(n, int)           #  random anomalies
    k = max(1, int(n * ar))
    for _ in range(k):
        st = np.random.randint(0, n - 50)
        tp = np.random.choice(['spk', 'shf', 'drp'])
        sid = np.random.choice(s)

        if tp == 'spk':  # sudden spike
            idx = st + np.random.randint(0, 50)
            df.at[idx, f's{sid+1}'] += np.random.uniform(8, 20)
            lbl[idx] = 1

        elif tp == 'shf':  
            ln = np.random.randint(10, 200)
            df.loc[st:st+ln, f's{sid+1}'] += np.random.uniform(-6, 6)
            lbl[st:st+ln+1] = 1

        else:  
            ln = np.random.randint(1, 20)
            df.loc[st:st+ln, f's{sid+1}'] = np.nan
            lbl[st:st+ln+1] = 1

    df['y'] = lbl
    return df


def clean_eda(df, scols):
    df = df.sort_values('time').reset_index(drop=True)          #  visualization
    df[scols] = df[scols].ffill().bfill().fillna(df[scols].median())

    print("Dataset shape:", df.shape)

    
    plt.figure(figsize=(12, 4))         # Plot sample 
    for c in scols:
        plt.plot(df['time'][:1000], df[c][:1000], label=c)
    plt.legend()
    plt.title("Sensor data (sample view)")
    plt.tight_layout()
    plt.savefig('plots/eda_preview.png')
    plt.show()

   
    plt.figure(figsize=(6, 4))       #  correlation heatmap 
    sns.heatmap(df[scols].corr(), annot=True, fmt=".2f")
    plt.title("Sensor correlation")
    plt.tight_layout()
    plt.savefig('plots/eda_corr.png')
    plt.show()

    return df


def feat_make(df, scols):                   # Feature engineering
    fdf = df.copy()
    for c in scols:
       
        for w in [5, 15, 60]:
            fdf[f'{c}_m{w}'] = fdf[c].rolling(w, 1).mean()
            fdf[f'{c}_s{w}'] = fdf[c].rolling(w, 1).std().fillna(0)

        
        for l in [1, 5, 15]:                # Lag features 
            fdf[f'{c}_l{l}'] = fdf[c].shift(l).bfill()

        
        fdf[f'{c}_t'] = fdf[c] - fdf[f'{c}_m5']             # Trend feature

    
    try:
        res = seasonal_decompose(df[scols[0]].values, period=1440, model='additive', extrapolate_trend='freq')
        fdf['sea'], fdf['res'] = res.seasonal, res.resid
    except:
        fdf['sea'], fdf['res'] = 0, 0

    return fdf.fillna(0)


def scale(x1, x2):
    sc = StandardScaler()
    return sc.fit_transform(x1), sc.transform(x2)


def iso_train(x):               #Isolation Forest
    m = IsolationForest(n_estimators=200, random_state=42)
    m.fit(x)
    return m


def ae_make(ts, fs, d=8):           # LSTM Autoencoder
    inp = layers.Input((ts, fs))
    e = layers.LSTM(64, return_sequences=True)(inp)
    e = layers.LSTM(d)(e)
    d1 = layers.RepeatVector(ts)(e)
    d1 = layers.LSTM(d, return_sequences=True)(d1)
    d1 = layers.LSTM(64, return_sequences=True)(d1)
    out = layers.TimeDistributed(layers.Dense(fs))(d1)
    m = models.Model(inp, out)
    m.compile(optimizer='adam', loss='mse')
    return m

def seq_make(x, l=30):
    return np.array([x[i:i+l] for i in range(len(x)-l+1)])



def eval_m(y, p):           # Evaluate model 
    p1, r1, f1, _ = precision_recall_fscore_support(y, p, average='binary', zero_division=0)
    return {'precision': float(p1), 'recall': float(r1), 'f1_score': float(f1)}


def plot_anom(df, col, mask, name):             # Plot anomalies detected 
    plt.figure(figsize=(14, 4))
    plt.plot(df['time'], df[col], label=col)
    plt.scatter(df.loc[mask, 'time'], df.loc[mask, col], s=10, c='r', label='Anomaly')
    plt.legend()
    plt.title(name)
    fn = f'plots/{name.lower().replace(" ", "_")}.png'
    plt.savefig(fn)
    plt.show()


def main():                        

    print(">> Generating synthetic data...")
    df = gen_data()
    scols = [c for c in df.columns if c.startswith('s')]

    print(">> Cleaning and EDA...")
    df = clean_eda(df, scols)

    print(">> Creating features...")
    df = feat_make(df, scols)
    fcols = [c for c in df.columns if any(s in c for s in scols)]

    
    i = int(len(df) * 0.7)      # Split data (70% train, 30% test)
    tr, te = df.iloc[:i], df.iloc[i:]
    x1, x2 = tr[fcols].values, te[fcols].values
    y2 = te['y'].values
    x1s, x2s = scale(x1, x2)

 
    print(">> Training Isolation Forest...")
    m1 = iso_train(x1s)
    p2 = (m1.predict(x2s) == -1).astype(int)
    e1 = eval_m(y2, p2)
    print("Isolation Forest metrics:", e1)
    plot_anom(te, scols[0], p2.astype(bool), "Isolation Forest")


    print(">> Training LSTM Autoencoder...")
    l = 30
    x1q, x2q = seq_make(x1s, l), seq_make(x2s, l)
    ts, fs = x1q.shape[1], x1q.shape[2]
    m2 = ae_make(ts, fs)
    es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    m2.fit(x1q, x1q, epochs=10, batch_size=128, validation_split=0.1, verbose=0, callbacks=[es])

  
    p2q = m2.predict(x2q)
    err = np.mean(np.mean((x2q - p2q) ** 2, axis=2), axis=1)
    thr = np.quantile(err, 0.995)
    p3 = (err > thr).astype(int)
    p3t = np.zeros(len(te), int)
    for i, v in enumerate(p3):
        if i + l - 1 < len(p3t):
            p3t[i + l - 1] = max(p3t[i + l - 1], v)

    e2 = eval_m(y2[l-1:], p3)
    print("LSTM Autoencoder metrics:", e2)
    plot_anom(te, scols[0], p3t.astype(bool), "LSTM Autoencoder")


    out = {'IsolationForest': e1, 'LSTM_AE': e2, 'Threshold': float(thr)}               # Save results
    with open('models/out.json', 'w') as f:
        json.dump(out, f, indent=2)
    print(">> Done. Results saved in models/out.json")

if __name__ == '__main__':
    main()




# This script builds an end-to-end anomaly detection system
# using synthetic IoT sensor data. It demonstrates data preparation,
# feature engineering, and two anomaly detection models:
# 1. Isolation Forest 
# 2. LSTM Autoencoder 