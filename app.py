from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)

# Load the trained XGBClassifier model
model = joblib.load("xgb_model.pkl")

# Features used during training
features = [
    'amount_paid', 'sender_bank', 'receiver_bank', 'receiving_currency',
    'payment_currency', 'payment_format', 'transaction_difference',
    'transaction_difference_percentage', 'log_amount_received', 'log_amount_paid',
    'rolling_mean_amount_7d', 'rolling_std_amount_7d', 'hour', 'day_of_week',
    'is_weekend', 'num_transactions_30d', 'avg_transaction_30d',
    'degree_centrality', 'pagerank_score', 'cross_currency_transaction',
    'time_diff', 'is_burst', 'z_score_amount', 'is_anomalous_amount',
    'is_circular', 'currency_arbitrage'
] + [f"gnn_embedding_{i}" for i in range(1, 17)]

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    ext = os.path.splitext(file.filename)[-1].lower()

    try:
        # Load file
        if ext == '.csv':
            df = pd.read_csv(file)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(file, engine='openpyxl')
        else:
            return jsonify({"error": "Unsupported file format"}), 400

        df.columns = df.columns.str.strip().str.lower()

        # Encode necessary categorical columns
        for col in ['sender_bank', 'receiver_bank', 'receiving_currency', 'payment_currency', 'payment_format']:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes

        # Check all required features are present
        missing = [col for col in features if col not in df.columns]
        if missing:
            return jsonify({"error": f"Missing required feature columns: {missing}"}), 400

        # Prepare input and predict
        X = df[features]
        preds = model.predict(X)

        # Build response
        results = []
        for i, row in df.iterrows():
            results.append({
                "transaction": i + 1,
                "from_bank": row.get("sender_bank", "N/A"),
                "to_bank": row.get("receiver_bank", "N/A"),
                "amount": float(row["amount_paid"]),
                "status": "Fraud" if preds[i] == 1 else "Legit"
            })

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": f"Backend error: {str(e)}"}), 500

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "API is working!"})

if __name__ == '__main__':
    app.run(debug=True)
