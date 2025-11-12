import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from flask import Flask, request, jsonify
from tensorflow.keras import layers, models, optimizers, regularizers

N_CLIENTS = 2   # Adjust as needed
NUM_ROUNDS = 5

#########################################
def load_server_test_data(path):
    df = pd.read_csv(path)
    df = df.replace({True: 1, False: 0})
    if 'label' not in df.columns:
        raise KeyError("CSV must include 'label' column")
    X = df.drop(columns=['label']).values.astype(float)
    y = df['label'].values.astype(int)
    return X, y

def print_server_metrics(model, X_test, y_test):
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs >= 0.5).astype(int).flatten()
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print("\nServer-side test set evaluation:")
    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred, digits=4))

SERVER_TEST_PATH = "../dataset/serverside_test.csv"  
X_test_server, y_test_server = load_server_test_data(SERVER_TEST_PATH)
########################################################################

KERAS_PARAMS = {
    "hidden_unit1": 64,
    "hidden_unit2": 64,
    "decay_rate": 1,
    "dropout": 0.4,
    "l2": 0.0001,
    "learning_rate": 0.001,
}

app = Flask(__name__)
current_round = 0
global_weights = None
global_weights_ready = False
global_weights_ready_count = 0
round_submissions = {}

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

def fedavg_aggregate_keras(client_models):
    valid = [c for c in client_models if c["n_samples"] > 0 and c["weights"] is not None]
    if not valid or len(valid) != N_CLIENTS:
        raise ValueError("Expected weights from all clients. Got: %d" % len(valid))
    sample_counts = [c["n_samples"] for c in valid]
    minority_counts = [c.get("n_minority", 0) for c in valid]
    total_samples = sum(sample_counts)
    total_minority = sum(minority_counts)
    n_weights = len(valid[0]["weights"])
    avg = [np.zeros_like(arr, dtype=np.float32) for arr in valid[0]["weights"]]
    weights = []
    for ns, nm in zip(sample_counts, minority_counts):
        w_samples = ns / total_samples if total_samples > 0 else 0.0
        w_minority = (nm / total_minority) if total_minority > 0 else 0.0
        w = 0.8 * w_minority + 0.2 * w_samples
        weights.append(w)
    w_sum = sum(weights) if sum(weights) > 0 else 1.0
    weights = [w / w_sum for w in weights]
    for client, w in zip(valid, weights):
        cw = client["weights"]
        for i in range(n_weights):
            avg[i] += cw[i] * w
    return avg

def serialize_weights(weights):
    return [w.tolist() for w in weights]

def deserialize_weights(weights_list):
    return [np.array(w) for w in weights_list]

@app.route('/initial_global_weights', methods=['GET'])
def initial_global_weights():
    global global_weights
    print("Server: Sent initial global weights to a client.")
    return jsonify({"weights": serialize_weights(global_weights)})
    
@app.route('/get_global_weights', methods=['GET'])
def get_global_weights():
    global global_weights_ready, global_weights_ready_count, global_weights
    if not global_weights_ready or global_weights is None:
        return jsonify({"ready": False, "weights": None})
    global_weights_ready_count += 1
    response = {"ready": True, "weights": serialize_weights(global_weights)}

    if global_weights_ready_count >= N_CLIENTS:
        global_weights_ready = False
        global_weights_ready_count = 0
        print(f"Server: All clients fetched aggregated global weights for Round {current_round}.")
    return jsonify(response)

@app.route('/upload_client_weights/<int:client_id>', methods=['POST'])
def upload_client_weights(client_id):
    global round_submissions, global_weights, current_round, global_weights_ready
    if not (0 <= client_id < N_CLIENTS):
        return jsonify({"status":"error","reason":"Invalid client ID"}), 400
    data = request.get_json()
    
    if not isinstance(data, dict) or "weights" not in data or "n_samples" not in data:
        return jsonify({"status":"error","reason":"Missing or malformed weights"}), 400
    if current_round not in round_submissions:
        round_submissions[current_round] = {}
    if client_id in round_submissions[current_round]:
        return jsonify({"status":"duplicate","reason":"Client already submitted in this round"}), 400

    weights = deserialize_weights(data["weights"])
    n_samples = data["n_samples"]
    n_minority = data.get("n_minority", 0)
    round_submissions[current_round][client_id] = {
        "client_id": client_id,
        "weights": weights,
        "n_samples": n_samples,
        "n_minority": n_minority
    }
    print(f"Server: Round {current_round}: Received weights from Client {client_id} ({n_samples} samples).")

    response = {"status": "received"}
    if len(round_submissions[current_round]) == N_CLIENTS:
        ordered_subs = [round_submissions[current_round][i] for i in range(N_CLIENTS)]
        global_weights = fedavg_aggregate_keras(ordered_subs)
        global_weights_ready = True
        print(f"Server: *** Aggregated Round {current_round}. Global weights ready for all clients. ***")
        current_round += 1
        response["status"] = "aggregated"
    if current_round >= NUM_ROUNDS:
        model = build_keras_model(input_dim, KERAS_PARAMS)
        model.set_weights(global_weights)
        print(f"\n=== FINAL PERFORMANCE after {NUM_ROUNDS} Rounds ===")
        print_server_metrics(model, X_test_server, y_test_server)
    return jsonify(response)

if __name__ == "__main__":
    input_dim = 52
    global_weights = build_keras_model(input_dim, KERAS_PARAMS).get_weights()
    global_weights_ready = True
    app.run(host='0.0.0.0', port=5000)