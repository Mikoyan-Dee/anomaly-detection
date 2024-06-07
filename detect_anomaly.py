import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder

data = np.array(['svchost.exe', 'svchost.exe', 'svchost.exe', 'svchost.exe', 'svchost.exe',
                 'svch0st.exe', 'svchost.exe'])


# Convert categorical data into numerical form
enc = LabelEncoder()
X = enc.fit_transform(data).reshape(-1, 1)

# Load Anomaly Detection Algorithm
isf = IsolationForest(contamination=0.1, random_state=62)
anomaly_indices = isf.fit_predict(X)

anomaly_results = np.where(anomaly_indices == -1)[0]
anomaly_output = np.array(data)[anomaly_results]
print("Detected anomalies:", anomaly_output)

