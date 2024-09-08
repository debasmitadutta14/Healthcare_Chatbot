from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import csv
import os
import datetime

app = Flask(__name__)

# Load the data and train the model
training = pd.read_csv('Training (1).csv')
cols = training.columns[:-1]  # Features
x = training[cols]
y = training['prognosis']
clf = DecisionTreeClassifier().fit(x, y)

severityDictionary = {}
description_list = {}
precautionDictionary = {}

# Load dictionaries
def load_dictionaries():
    global severityDictionary, description_list, precautionDictionary

    # Load Symptom Severity
    with open('Symptom_severity.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for rows in reader:
            if len(rows) >= 2:
                severityDictionary[rows[0]] = int(rows[1])

    # Load Symptom Descriptions
    with open('symptom_Description.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for rows in reader:
            if len(rows) >= 2:
                description_list[rows[0]] = rows[1]

    # Load Precautions
    with open('symptom_precaution.csv') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for rows in reader:
            if len(rows) >= 5:
                precautionDictionary[rows[0]] = rows[1:5]

load_dictionaries()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms = request.json.get('symptoms', [])
    days = int(request.json.get('days', 0))

    # Initialize input vector
    input_vector = np.zeros(len(cols))

    # Set the input vector for the symptoms provided
    for symptom in symptoms:
        if symptom in cols.values:
            input_vector[cols.get_loc(symptom)] = 1

    # Predict the disease using the decision tree
    tree_prediction = clf.predict([input_vector])[0]

    # Calculate severity
    severity_sum = sum([severityDictionary.get(symptom, 0) for symptom in symptoms])

    # Prepare the response
    response = {
        'disease': tree_prediction,
        'description': description_list.get(tree_prediction, 'No description available.'),
        'precautions': precautionDictionary.get(tree_prediction, []),
        'doctor_consultation': severity_sum * days / (len(symptoms) + 1) > 13
    }

    return jsonify(response)

@app.route('/schedule', methods=['POST'])
def schedule_appointment():
    data = request.json
    name = data.get('name')
    doctor_name = data.get('doctor_name')
    location = data.get('location')
    date_input = data.get('date')
    time_input = data.get('time')

    try:
        appointment_date = datetime.datetime.strptime(date_input, "%Y-%m-%d").date()
        appointment_time = datetime.datetime.strptime(time_input, "%H:%M").time()
        



        with open('appointments.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, doctor_name, location, appointment_date, appointment_time])

        return jsonify({'status': 'success', 'message': f"Appointment scheduled for {name} with  {doctor_name} at {location} on {appointment_date} at {appointment_time}."})
    except ValueError:
        return jsonify({'status': 'error', 'message': 'Invalid date or time format.'}), 400

@app.route('/appointments', methods=['GET'])
def get_appointments():
    appointments = []
    if os.path.exists('appointments.csv'):
        with open('appointments.csv', mode='r') as file:
            reader = csv.reader(file)
            appointments = [{'name': row[0], 'doctor': row[1], 'location': row[2], 'date': row[3], 'time': row[4]} for row in reader]

    return jsonify(appointments)

if __name__ == "__main__":
    app.run(debug=True)
