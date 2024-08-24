from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load your trained model
model = pickle.load(open('baku_apartment_price_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    square = float(request.form['square'])
    new_building = int(request.form['new_building'])
    has_repair = int(request.form['has_repair'])
    has_metro = int(request.form['has_metro'])
    
    # Reset location features to 0
    location_features = {
        'location_Binəqədi': 0,
        'location_Lökbatan_q': 0,
        'location_Nizami': 0,
        'location_Nərimanov': 0,
        'location_Nəsimi': 0,
        'location_Pirallahı_r': 0,
        'location_Qaradağ': 0,
        'location_Sabunçu': 0,
        'location_Suraxanı': 0,
        'location_Səbail': 0,
        'location_Xətai': 0,
        'location_Xəzər': 0,
        'location_Yasamal': 0,
    }
    
    # Set the selected location to 1
    selected_location = request.form['location']
    location_features[selected_location] = 1
    
    # Prepare the features for prediction
    features = [square, new_building, has_repair, has_metro] + list(location_features.values())
    
    # Make a prediction
    prediction = model.predict([features])
    output = round(prediction[0], 2)
    
    return render_template('index.html', prediction_text=f'Predicted Price: {output} AZN')

if __name__ == "__main__":
    app.run(debug=True)
