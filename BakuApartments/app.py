from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes

# Load the ML model
with open('baku_apartment_price_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = [
        data['square'],
        data['new_building'],
        data['has_repair'],
        data['has_metro'],
        data['location_Binəqədi'],
        data['location_Lökbatan q.'],
        data['location_Nizami'],
        data['location_Nərimanov'],
        data['location_Nəsimi'],
        data['location_Pirallahı r.'],
        data['location_Qaradağ'],
        data['location_Sabunçu'],
        data['location_Suraxanı'],
        data['location_Səbail'],
        data['location_Xətai'],
        data['location_Xəzər'],
        data['location_Yasamal']
    ]
    
    prediction = model.predict([features])[0]
    return jsonify({'price': prediction})

if __name__ == '__main__':
    app.run(debug=True)
