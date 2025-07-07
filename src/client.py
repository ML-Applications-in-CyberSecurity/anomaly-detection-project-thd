import socket
import json
import pandas as pd
import joblib
from together import Together
from environs import Env
from datetime import datetime, timezone
import os


env = Env()
env.read_env()
HOST = 'localhost'
PORT = 9999

model = joblib.load("anomaly_model.joblib")
scaler = joblib.load("confidence_scaler.joblib")

# Initialize Together.ai client
client = Together(api_key=env("API_KEY"))
LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free" # DeepSeek LLaMA B70 3LLM equivalent

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

                processed_data = pre_process_data(data)
                prediction = model.predict(processed_data)[0]
                score_raw = model.decision_function(processed_data)[0]  # raw score
                score_scaled = scaler.transform([[score_raw]])[0][0]    # normalized between 0 and 1

                if prediction == -1:
                    print("Anomaly Detected")
                    print(f"Confidence Score: {score_scaled:.2f}")

                    messages = [
                        {"role": "system", "content": "You are a helpful assitant that labels sensor anomalies."},
                        {"role": "user", "content": f"Sensor reading: {data}\nDescribe the type of anomaly and suggest a possible cause."}
                    ]
                    response = client.chat.completions.create(
                        model=LLM_MODEL,
                        messages=messages,
                        temperature=0.7,
                        max_tokens=300,
                        stream=False
                    )
                    label_response = response.choices[0].message.content.strip()
                    # Prepare anomaly record
                    anomaly_record = pd.DataFrame([{
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "src_port": data["src_port"],
                        "dst_port": data["dst_port"],
                        "packet_size": data["packet_size"],
                        "duration_ms": data["duration_ms"],
                        "protocol": data["protocol"],
                        "confidence_score": score_scaled,
                        "llm_label": label_response
                    }])
                    # Path to csv file
                    csv_file = "anomalies_log.csv"
                    # If file exists, appand without header; otherwise, write with header
                    if os.path.isfile(csv_file):
                        anomaly_record.to_csv(csv_file, mode='a', header=False, index=False)
                    else:
                        anomaly_record.to_csv(csv_file, mode='w', header=True, index=False)
                    print(f"\n Anomaly Details: \n{label_response}\n")
                else:
                    print("Normal Traffic. \n")
                    print(f"Confidence Score: {score_scaled:.2f}")
                print("_" * 70)
            except json.JSONDecodeError:
                print("Error decoding JSON.")
