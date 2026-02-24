import os
import pandas as pd
import torch

from src.models.lstm_latent import LatentLSTM

# ==============================
# CONFIG
# ==============================
LATENT_DIR = "data_processed/latent"
MODEL_PATH = "models/lstm_latent.pt"
OUT_PATH = "data_processed/risk_signal.csv"

SEQ_LEN = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(input_dim):
    model = LatentLSTM(input_dim=input_dim)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def run():
    sample_file = os.listdir(LATENT_DIR)[0]
    sample_df = pd.read_csv(os.path.join(LATENT_DIR, sample_file))
    latent_cols = [c for c in sample_df.columns if c.startswith("latent")]
    input_dim = len(latent_cols)

    model = load_model(input_dim)

    os.makedirs("data_processed/risk", exist_ok=True)

    for file in os.listdir(LATENT_DIR):
        if not file.endswith(".csv"):
            continue

        symbol = file.replace("_latent.csv", "")
        print(f"🔍 Risk inference: {symbol}")

        df = pd.read_csv(os.path.join(LATENT_DIR, file))
        df["date"] = pd.to_datetime(df["date"], format="mixed")

        data = df[latent_cols].values
        rows = []

        for i in range(SEQ_LEN, len(data)):
            x_seq = torch.tensor(
                data[i - SEQ_LEN:i],
                dtype=torch.float32
            ).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                raw_risk = model(x_seq).item()

            rows.append({
                "date": df.loc[i, "date"],
                "raw_risk": raw_risk
            })

        df_risk = pd.DataFrame(rows)

        # Z-score normalization
        df_risk["risk_score"] = (
            df_risk["raw_risk"] - df_risk["raw_risk"].rolling(60).mean()
        ) / df_risk["raw_risk"].rolling(60).std()

        df_risk = df_risk.dropna()[["date", "risk_score"]]

        out_path = f"data_processed/risk/{symbol}_risk.csv"
        df_risk.to_csv(out_path, index=False)

        print(f" Saved → {out_path}")


if __name__ == "__main__":
    run()
 