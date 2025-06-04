# retrain.py
import subprocess

print("📦 Retraining CNN model...")
subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "models/train_cnn.ipynb", "--inplace"])

print("📦 Retraining XGBoost model...")
subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", "models/train_xgb.ipynb", "--inplace"])

print("✅ Retraining done!")
