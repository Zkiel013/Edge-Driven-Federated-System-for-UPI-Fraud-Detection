Markdown
# Edge-Driven Federated System for UPI Fraud Detection

A privacy-preserving machine learning framework deployed on AWS EC2 that detects fraudulent UPI transactions using Federated Learning. By decentralizing the training process, transaction data remains on the edge device, and only model updates are shared.

## Project Structure

client/
├── client1.py
├── client2.py
└── requirements.txt
server/
├── server.py
└── requirements.txt
images/
├── architecture.png
└── roc_curve.png


## Features

- Privacy-preserving edge-driven model training
- Federated Learning using the FedAvg algorithm
- Machine Learning-based prediction of fraudulent UPI transactions
- Handles highly imbalanced financial datasets
- AWS EC2 deployment simulating a real-world distributed network
- Secure client-server communication using Flask

## Tech Stack

| Tech             | Description                       |
|------------------|------------------------------------|
| Python 3         | Core programming language         |
| TensorFlow       | ML model building and training    |
| Flask            | Central server communication API  |
| Scikit-Learn     | Data preprocessing and metrics    |
| AWS EC2          | Cloud infrastructure deployment   |
| Imbalanced-Learn | Handling skewed dataset classes   |

## Setup Instructions

### 1. Clone the repository

```bash
git clone [https://github.com/Zkiel013/Edge-Driven-Federated-System-for-UPI-Fraud-Detection.git](https://github.com/Zkiel013/Edge-Driven-Federated-System-for-UPI-Fraud-Detection.git)
cd Edge-Driven-Federated-System-for-UPI-Fraud-Detection
2. Install dependencies
For server:

Bash
cd server
pip install flask numpy pandas scikit-learn tensorflow
For client:

Bash
cd client
pip install numpy pandas scikit-learn imbalanced-learn tensorflow requests
3. Set up environment configurations
Open the client scripts (client1.py and client2.py) and update the SERVER_URL to point to your server's public IP address:

Python
# Inside client scripts
SERVER_URL = "http://<Server-Public-IP>:5000"
4. Run the app
Bash
# In one terminal (Central Server)
cd server
python3 server.py

# In another terminal (Client 1 Node)
cd client
python3 client1.py

# In a third terminal (Client 2 Node)
cd client
python3 client2.py
App will communicate over:

Server Port: 5000

ML Model
5-layer Multi-Layer Perceptron (MLP) architecture

Model trained locally on client nodes with private data (98.5% legitimate, 1.5% fraudulent)

Weight aggregation using Federated Averaging (FedAvg) over multiple rounds

Optimized using Adam Optimizer and Binary Cross-Entropy loss

Evaluated using AUC-ROC, Precision, and Weighted Recall

🖼️ Project Screenshots
Contributing
Fork the repository

Create a new branch

Commit your changes

Push to your branch

Create a Pull Request
