from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import random
import json

app = Flask(__name__)

# Generate random housing data
def generate_random_data(num_samples=1000):
    np.random.seed(42)
    
    data = {
        'size_sqft': np.random.randint(800, 4000, num_samples),
        'bedrooms': np.random.randint(1, 6, num_samples),
        'bathrooms': np.random.randint(1, 4, num_samples),
        'year_built': np.random.randint(1950, 2023, num_samples),
        'location_score': np.random.uniform(1, 10, num_samples),
        'has_garage': np.random.choice([0, 1], num_samples),
        'distance_to_city': np.random.uniform(1, 30, num_samples)
    }
    
    # Create price based on features with some randomness
    base_price = (
        data['size_sqft'] * 150 + 
        data['bedrooms'] * 20000 + 
        data['bathrooms'] * 15000 +
        (2023 - data['year_built']) * -500 +
        data['location_score'] * 10000 +
        data['has_garage'] * 25000 -
        data['distance_to_city'] * 3000
    )
    
    # Add some noise
    data['price'] = base_price + np.random.normal(0, 50000, num_samples)
    
    return pd.DataFrame(data)

# Generate and prepare data
housing_data = generate_random_data(1000)
X = housing_data.drop('price', axis=1)
y = housing_data['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()
        
        # Prepare features for prediction
        features = np.array([[
            data['size_sqft'],
            data['bedrooms'],
            data['bathrooms'],
            data['year_built'],
            data['location_score'],
            data['has_garage'],
            data['distance_to_city']
        ]])
        
        # Make prediction
        prediction = model.predict(features)
        
        # Return prediction
        return jsonify({
            'predicted_price': round(float(prediction[0]), 2),
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        features_list = data['houses']
        
        predictions = []
        for house in features_list:
            features = np.array([[
                house['size_sqft'],
                house['bedrooms'],
                house['bathrooms'],
                house['year_built'],
                house['location_score'],
                house['has_garage'],
                house['distance_to_city']
            ]])
            
            prediction = model.predict(features)
            predictions.append(round(float(prediction[0]), 2))
        
        return jsonify({
            'predictions': predictions,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'})

@app.route('/model_info')
def model_info():
    # Get feature importance
    feature_importance = dict(zip(X.columns, model.feature_importances_))
    
    return jsonify({
        'feature_importance': feature_importance,
        'training_score': round(model.score(X_train, y_train), 4),
        'test_score': round(model.score(X_test, y_test), 4),
        'num_samples': len(housing_data)
    })

@app.route('/sample_data')
def sample_data():
    # Return some sample data for testing
    sample = housing_data.sample(5)
    return jsonify({
        'sample_data': sample.to_dict('records')
    })

if __name__ == '__main__':
    print("Starting Flask server...")
    print(f"Model trained with {len(housing_data)} samples")
    print(f"Training score: {model.score(X_train, y_train):.4f}")
    print(f"Test score: {model.score(X_test, y_test):.4f}")
    app.run(debug=True, host='0.0.0.0', port=80)
