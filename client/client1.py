import numpy as np
import pandas as pd
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
import requests
from tensorflow.keras import layers, models, optimizers, regularizers, callbacks

#SERVER_URL = "http://127.0.0.1:5000"
SERVER_URL = "ec2-15-207-112-181.ap-south-1.compute.amazonaws.com:5000"
CLIENT_ID = 0   
DATA_PATH = "../dataset/client1_dataset.csv" 
NUM_ROUNDS = 5
KERAS_PARAMS = {
    "hidden_unit1": 64,
    "hidden_unit2": 64,
    "decay_rate": 1,
    "dropout": 0.4,
    "l2": 0.0001,
    "learning_rate": 0.001,
    "local_epochs": 200,
    "batch_size": 32,
    "patience": 2
}
SMOTE_DEFAULT_K = 5

def build_keras_model(input_dim, params):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(params["hidden_unit1"], activation="relu", kernel_regularizer=regularizers.l2(params["l2"]))(inp)
    x = layers.Dropout(params["dropout"])(x)
    x = layers.Dense(params["hidden_unit2"], activation="relu", kernel_regularizer=regularizers.l2(params["l2"]))(x)
    x = layers.Dropout(params["dropout"])(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inp, outputs=out)
    opt = optimizers.Adam(learning_rate=params["learning_rate"])
    model.compile(optimizer=opt, loss="binary_crossentropy")
    return model

def serialize_weights(weights):
    return [w.tolist() for w in weights]

def deserialize_weights(weights_list):
    return [np.array(w) for w in weights_list]

def load_data(path):
    df = pd.read_csv(path)
    df = df.replace({True: 1, False: 0})
    if 'label' not in df.columns:
        raise KeyError("CSV must include 'label' column")
    df['label'] = df['label'].astype(int)
    X = df.drop(columns=['label']).values.astype(float)
    y = df['label'].values.astype(int)
    return X, y
def fetch_initial_global_weights():
    while True:
        try:
            resp = requests.get(f"{SERVER_URL}/initial_global_weights", timeout=15)
        except Exception as e:
            print("Network/server error (initial):", str(e))
            import time; time.sleep(2)
            continue
        data = resp.json()
        if data.get("weights") is not None:
            return deserialize_weights(data["weights"])
        else:
            print("Waiting for initial global weights from server...")
            import time; time.sleep(2)

def fetch_global_weights():
    while True:
        try:
            resp = requests.get(f"{SERVER_URL}/get_global_weights", timeout=15)
        except Exception as e:
            print(f"Client {CLIENT_ID}: Network/server error:", str(e))
            import time; time.sleep(2)
            continue
        data = resp.json()
        if data.get("ready"):
            return deserialize_weights(data["weights"])
        else:
            print(f"Client {CLIENT_ID}: Awaiting server aggregation for next round...")
            import time; time.sleep(2)

def upload_weights(weights, n_samples, n_minority):
    payload = {
        "weights": serialize_weights(weights),
        "n_samples": n_samples,
        "n_minority": n_minority
    }
    while True:
        try:
            resp = requests.post(f"{SERVER_URL}/upload_client_weights/{CLIENT_ID}", json=payload, timeout=15)
        except Exception as e:
            print("Server upload error:", str(e))
            import time; time.sleep(2)
            continue
        try:
            out = resp.json()
        except Exception:
            out = {"status":"error"}
        if out.get("status") in ["received", "aggregated"]:
            return out
        print("Upload error/reject:", out)
        import time; time.sleep(5)

def client_train_keras(client_id, X_local, y_local, global_weights=None):
    if X_local.shape[0] == 0:
        print("Warning: no samples for training")
        return {"client_id": client_id, "n_samples": 0, "n_minority": 0, "weights": None}
    cnt = Counter(y_local)
    minority_count = cnt.get(1, 0)
    if minority_count == 0:
        X_res, y_res = X_local, y_local
    elif minority_count == 1:
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_local, y_local)
    else:
        k = min(SMOTE_DEFAULT_K, max(1, minority_count - 1))
        sm = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = sm.fit_resample(X_local, y_local)
    X_res = X_res.astype(np.float32)
    y_res = y_res.astype(np.float32)
    input_dim = X_res.shape[1]
    minority_after = int(np.sum(y_res == 1))
    model = build_keras_model(input_dim, KERAS_PARAMS)
    if global_weights is not None:
        try:
            model.set_weights(global_weights)
        except Exception:
            print("Warning: weights shape mismatch, resetting model weights.")
    class_counts = np.bincount(y_res.astype(int), minlength=2)
    total = len(y_res)
    sample_weight = None
    if class_counts.sum() > 0:
        cw = {i: (total / (2.0 * c)) if c > 0 else 1.0 for i, c in enumerate(class_counts)}
        sample_weight = np.array([cw[int(y)] for y in y_res], dtype=float)
    cb = [callbacks.EarlyStopping(monitor="loss", patience=KERAS_PARAMS["patience"], restore_best_weights=True)]
    model.fit(X_res, y_res, epochs=KERAS_PARAMS["local_epochs"], batch_size=KERAS_PARAMS["batch_size"],
              verbose=0, sample_weight=sample_weight, callbacks=cb)
    client_weights = model.get_weights()
    return {"client_id": client_id, "n_samples": X_res.shape[0], "n_minority": minority_after, "weights": client_weights}

if __name__ == "__main__":
    X_local, y_local = load_data(DATA_PATH)
    rounds = NUM_ROUNDS

    global_weights = fetch_initial_global_weights()

    r = 0
    while r < rounds:
        print(f"\n--- Client {CLIENT_ID}: Starting Round {r+1} ---")
        print(f"Client {CLIENT_ID}: Received global weights for Round {r+1}, now training on local data...")
        results = client_train_keras(CLIENT_ID, X_local, y_local, global_weights)
        out = upload_weights(results["weights"], results["n_samples"], results["n_minority"])
        if out.get('status') == "aggregated":
            print(f"Client {CLIENT_ID}: Upload complete. Server has aggregated and is ready for the next round.")
        elif out.get('status') == "received":
            print(f"Client {CLIENT_ID}: Uploaded local weights, waiting for other clients to upload.")
        elif out.get('status') == "duplicate":
            print(f"Client {CLIENT_ID}: Already uploaded for this round, waiting for aggregation.")
        else:
            print(f"Client {CLIENT_ID}: Received server status: {out.get('status')} ({out.get('reason', '')})")

        print(f"Client {CLIENT_ID}: Waiting for new global weights from server for next round...")
        global_weights = None
        while global_weights is None:
            fetched = fetch_global_weights()
            if fetched is not None:
                global_weights = fetched
        print(f"Client {CLIENT_ID}: Received new global weights, moving to next round.")
        r += 1
    print(f"\nClient {CLIENT_ID}: Completed all {rounds} rounds of federated training.")
