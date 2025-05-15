from flask import Flask, jsonify, request, send_file, send_from_directory
from flask_cors import CORS
import numpy as np
from scipy.signal import butter, filtfilt
import os
import json
import time
from datetime import datetime
import uuid

app = Flask(__name__, static_folder="static")
CORS(app)

# Data storage for patient reports
REPORTS_DIR = "patient_reports"
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# Default settings
DEFAULT_SETTINGS = {
    "sampling_rate": 300,
    "lowcut": 0.05,
    "highcut": 40,
    "anomaly_threshold": 0.7,
    "simulation_speed": 1.0
}

# Try to load settings from file
SETTINGS_FILE = "settings.json"
if os.path.exists(SETTINGS_FILE):
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
    except:
        settings = DEFAULT_SETTINGS.copy()
else:
    settings = DEFAULT_SETTINGS.copy()
    with open(SETTINGS_FILE, 'w') as f:
        json.dump(settings, f)

# Try to load model, but don't stop if it fails
try:
    if os.path.exists('heartbeat_model_xyz.h5'):
        from tensorflow.keras.models import load_model
        model = load_model('heartbeat_model_xyz.h5')
        print("Model loaded successfully!")
    else:
        print("Model file not found, using mock predictions for testing")
        model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Bandpass filter function to process ECG signal
def bandpass_filter(signal, lowcut=settings['lowcut'], highcut=settings['highcut'], fs=settings['sampling_rate'], order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

# Normalize signal
def normalize(signal):
    if np.max(signal) == np.min(signal):
        return np.zeros_like(signal)
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 2 - 1

# Function to generate simulated ECG data
def generate_simulated_ecg(duration=10, fs=settings['sampling_rate'], with_anomaly=False):
    # Basic parameters for normal ECG
    heart_rate = np.random.uniform(60, 100)  # beats per minute
    t = np.arange(0, duration, 1.0/fs)
    
    # Generate basic signal
    signal = np.zeros_like(t)
    
    # Base frequency components (simplified)
    for i in range(len(t)):
        # Basic heart rhythm
        phase = (heart_rate / 60) * 2 * np.pi * t[i]
        
        # P wave (atrial depolarization)
        signal[i] += 0.25 * np.exp(-((t[i] % (60/heart_rate) - 0.2) ** 2) / 0.001)
        
        # QRS complex (ventricular depolarization)
        qrs_time = t[i] % (60/heart_rate)
        if 0.3 <= qrs_time <= 0.4:
            # Q wave
            signal[i] -= 0.05 * np.exp(-((qrs_time - 0.3) ** 2) / 0.0001)
        elif 0.4 < qrs_time <= 0.45:
            # R wave
            signal[i] += 1.0 * np.exp(-((qrs_time - 0.425) ** 2) / 0.0002)
        elif 0.45 < qrs_time <= 0.5:
            # S wave
            signal[i] -= 0.3 * np.exp(-((qrs_time - 0.475) ** 2) / 0.0001)
        
        # T wave (ventricular repolarization)
        signal[i] += 0.3 * np.exp(-((t[i] % (60/heart_rate) - 0.7) ** 2) / 0.004)
    
    # Add some noise
    noise = np.random.normal(0, 0.05, len(t))
    signal += noise
    
    # Add anomaly if requested
    if with_anomaly:
        # Determine random position for anomaly
        anomaly_pos = int(np.random.uniform(0.2, 0.8) * len(signal))
        anomaly_length = int(fs * 0.5)  # Half a second anomaly
        
        # Types of anomalies
        anomaly_type = np.random.choice(['extra_beat', 'missing_beat', 'irregular_beat'])
        
        if anomaly_type == 'extra_beat':
            # Insert an extra beat
            for i in range(anomaly_length):
                if anomaly_pos + i < len(signal):
                    pos = anomaly_pos + i
                    qrs_time = (i / anomaly_length) * 0.3
                    if qrs_time <= 0.1:
                        signal[pos] -= 0.1 * np.exp(-((qrs_time - 0.05) ** 2) / 0.0001)
                    elif 0.1 < qrs_time <= 0.2:
                        signal[pos] += 1.5 * np.exp(-((qrs_time - 0.15) ** 2) / 0.0002)
                    elif 0.2 < qrs_time <= 0.3:
                        signal[pos] -= 0.4 * np.exp(-((qrs_time - 0.25) ** 2) / 0.0001)
        
        elif anomaly_type == 'missing_beat':
            # Flatten the signal as if a beat is missing
            if anomaly_pos + anomaly_length < len(signal):
                signal[anomaly_pos:anomaly_pos+anomaly_length] = np.mean(signal) + noise[anomaly_pos:anomaly_pos+anomaly_length] * 0.5
        
        else:  # irregular_beat
            # Create an irregular beat pattern
            if anomaly_pos + anomaly_length < len(signal):
                # Amplify and distort a section
                signal[anomaly_pos:anomaly_pos+anomaly_length] *= 1.5
                signal[anomaly_pos:anomaly_pos+anomaly_length] += np.sin(np.linspace(0, 10*np.pi, anomaly_length)) * 0.3
    
    return signal.tolist()

# Function to process incoming ECG data and predict
def predict_ecg_signal(signal):
    try:
        # Ensure signal is numpy array
        signal = np.array(signal, dtype=np.float32)
        
        # Apply bandpass filter
        filtered_signal = bandpass_filter(signal)
        
        # Normalize the ECG signal
        normalized_signal = normalize(filtered_signal)
        
        # Ensure signal is in the correct shape (1000, 1)
        if len(normalized_signal) < 1000:
            # Pad if too short
            normalized_signal = np.pad(normalized_signal, (0, 1000 - len(normalized_signal)), 'constant')
        else:
            # Take first 1000 points if too long
            normalized_signal = normalized_signal[:1000]
        
        reshaped_signal = normalized_signal.reshape(1, 1000, 1)
        
        # Use model if available, otherwise return mock prediction
        if model is not None:
            # If using actual model with dual inputs (CNN+LSTM)
            prediction = model.predict([reshaped_signal, reshaped_signal])
            return float(prediction[0][0])
        else:
            # For testing, return random value with occasional high values
            return np.random.choice([np.random.random() * 0.3, np.random.random() * 0.7 + 0.3, np.random.random() * 0.2 + 0.8], 
                                   p=[0.7, 0.2, 0.1])
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 0.0

# API endpoint for real-time prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        signal = data['signal']
        
        # Predict the anomaly score from the ECG signal
        prediction = predict_ecg_signal(signal)
        
        # Return the prediction (anomaly score) and processed signal for visualization
        return jsonify({
            'prediction': prediction,
            'threshold': settings['anomaly_threshold'],
            'status': 'danger' if prediction > settings['anomaly_threshold'] else 'normal'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# API endpoint to generate simulated ECG data
@app.route('/simulate', methods=['GET'])
def simulate():
    try:
        # Get parameters from query string
        duration = float(request.args.get('duration', 10))
        with_anomaly = request.args.get('anomaly', 'false').lower() == 'true'
        
        # Generate simulated ECG data
        signal = generate_simulated_ecg(duration=duration, with_anomaly=with_anomaly)
        
        return jsonify({
            'signal': signal,
            'duration': duration,
            'sampling_rate': settings['sampling_rate']
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# API endpoint to save patient report
@app.route('/save_report', methods=['POST'])
def save_report():
    try:
        data = request.json
        patient_name = data.get('patient_name', 'Unknown')
        patient_id = data.get('patient_id', str(uuid.uuid4())[:8])
        monitoring_duration = data.get('duration', 0)
        anomalies_detected = data.get('anomalies_detected', 0)
        avg_heart_rate = data.get('avg_heart_rate', 0)
        notes = data.get('notes', '')
        timestamps = data.get('timestamps', [])
        predictions = data.get('predictions', [])
        
        # Create report
        report = {
            'patient_name': patient_name,
            'patient_id': patient_id,
            'report_id': str(uuid.uuid4()),
            'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'monitoring_duration': monitoring_duration,
            'anomalies_detected': anomalies_detected,
            'avg_heart_rate': avg_heart_rate,
            'notes': notes,
            'data': {
                'timestamps': timestamps,
                'predictions': predictions
            }
        }
        
        # Save report to file
        report_filename = f"{report['report_id']}.json"
        with open(os.path.join(REPORTS_DIR, report_filename), 'w') as f:
            json.dump(report, f, indent=2)
        
        return jsonify({
            'success': True,
            'report_id': report['report_id'],
            'message': 'Report saved successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# API endpoint to get list of reports
@app.route('/reports', methods=['GET'])
def get_reports():
    try:
        reports = []
        if os.path.exists(REPORTS_DIR):
            for filename in os.listdir(REPORTS_DIR):
                if filename.endswith('.json'):
                    with open(os.path.join(REPORTS_DIR, filename), 'r') as f:
                        report = json.load(f)
                        # Add summary information only
                        reports.append({
                            'report_id': report['report_id'],
                            'patient_name': report['patient_name'],
                            'patient_id': report['patient_id'],
                            'date': report['date'],
                            'anomalies_detected': report['anomalies_detected']
                        })
        
        return jsonify({
            'reports': reports
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# API endpoint to get a specific report
@app.route('/reports/<report_id>', methods=['GET'])
def get_report(report_id):
    try:
        report_file = os.path.join(REPORTS_DIR, f"{report_id}.json")
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                report = json.load(f)
            return jsonify(report)
        else:
            return jsonify({'error': 'Report not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# API endpoint to get current settings
@app.route('/settings', methods=['GET'])
def get_settings():
    return jsonify(settings)

# API endpoint to update settings
@app.route('/settings', methods=['POST'])
def update_settings():
    try:
        data = request.json
        # Update settings
        for key, value in data.items():
            if key in settings:
                settings[key] = value
        
        # Save settings to file
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings, f, indent=2)
        
        return jsonify({
            'success': True,
            'settings': settings
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Simple health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok'})

# Serve the HTML file
@app.route('/')
def index():
    return send_file('index.html')

# Serve static files
@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("Starting ECG monitoring server...")
    app.run(debug=True, host="0.0.0.0", port=5000)