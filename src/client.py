import socket
import json
import pandas as pd
import joblib
from environs import Env


env = Env()
env.read_env()
HOST = 'localhost'
PORT = 9999

model = joblib.load("anomaly_model.joblib")
together.api_key = env("API_KEY")

def pre_process_data(data):
    # Convert data to DataFrame for model prediction
    df = pd.DataFrame([data])
    # One-hot encode protocol column
    df = pd.get_dummies(df, columns=['protocol'], drop_first=True)
    # If the protocol is only 'TCP', the 'protocol_UDP' column won't be created
    # We add it manually to match the training data structure
    if 'protocol_UDP' not in df.columns:
        df['protocol_UDP'] = 0
    # Sorting columns
    df = df[['src_port', 'dst_port', 'packet_size', 'duration_ms', 'protocol_UDP']]
    return df.astype(int).to_numpy()

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))
    buffer = ""
    print("Client connected to server.\n")

    while True:
        chunk = s.recv(1024).decode()
        if not chunk:
            break
        buffer += chunk

        while '\n' in buffer:
            line, buffer = buffer.split('\n', 1)
            try:
                data = json.loads(line)
                print(f'Data Received:\n{data}\n')

                #TODO 3: Here you have to add code to process the received data and detect anomalies using a trained model.

                #TODO 4: Here you have to connect to a LLM using together ai with your api code to caption the alert for data and anomalies detected.

            except json.JSONDecodeError:
                print("Error decoding JSON.")
