import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import pickle

# Load data
df = pd.read_csv("HI-Small_Trans.csv")
df.columns = df.columns.str.strip().str.lower()

# Rename for consistency
col_renames = {
    'timestamp': 'TimeStamp',
    'frombank': 'FromBank',
    'tobank': 'ToBank',
    'to bank': 'ToBank',
    'account': 'Account',
    'toaccount': 'ToAccount',
    'amount received': 'Amount Received',
    'received currency': 'Received Currency',
    'amount paid': 'Amount Paid',
    'payment currency': 'Payment Currency',
    'payment format': 'Payment Format',
    'is laundering': 'Is Laundering'
}
df.rename(columns=col_renames, inplace=True)

# Parse time
def get_seconds(ts):
    try:
        parts = str(ts).split()[1].split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60
    except:
        return 0

df["ParsedTime"] = df["TimeStamp"].apply(get_seconds)

# Encode categoricals
categorical_cols = ["Received Currency", "Payment Currency", "Payment Format"]
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# ✅ Use actual target column
if "Is Laundering" not in df.columns:
    raise ValueError("Missing target column: Is Laundering")
df["is_fraud"] = df["Is Laundering"].astype(int)

# Features used to train the model
features = [
    "Amount Paid",
    "ParsedTime",
    "FromBank",
    "ToBank",
    "Received Currency",
    "Payment Currency",
    "Payment Format"
]

# Ensure all features exist
for col in features:
    if col not in df.columns:
        raise ValueError(f"Missing feature column: {col}")

X = df[features]
y = df["is_fraud"]

# Train and save
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

with open("model_rndf.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained with real labels and saved!")
