import sys
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import json
import traceback

try:
    # Load model
    model = xgb.Booster()
    model.load_model("xgb_model.json")

    # Load input file
    file_path = sys.argv[1]
    original_df = pd.read_excel(file_path, engine='openpyxl')
    original_df.fillna(0, inplace=True)
    df = original_df.copy()

    # Drop non-feature columns
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or col.lower() == 'is_laundering':
            df.drop(columns=[col], inplace=True)

    # Encode categorical features
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str)
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

    # Keep only numeric types
    valid_dtypes = ['int64', 'float64', 'int32', 'float32', 'bool']
    df = df[[col for col in df.columns if df[col].dtype.name in valid_dtypes]]

    # Predict
    dX = xgb.DMatrix(df)
    probs = model.predict(dX)
    labels = (probs >= 0.85).astype(int)

    # Build result table (clean output)
    results = []
    for i in range(len(original_df)):
        results.append({
            "transaction": i + 1,
            "from_account": str(original_df.get("Sender_Account", ["N/A"])[i]),
            "to_account": str(original_df.get("Receiver_Account", ["N/A"])[i]),
            "timestamp": str(original_df.get("Timestamp", ["N/A"])[i]),
            "currency": str(original_df.get("Payment_Currency", ["N/A"])[i]),
            "status": "Fraud" if labels[i] else "Legit"
        })

    print(json.dumps(results))

except Exception:
    print("❌ Python Error:\n" + traceback.format_exc(), file=sys.stderr)
    sys.exit(1)
