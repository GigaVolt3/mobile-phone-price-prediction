from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import os

app = Flask(__name__)

# Global variables
df = None
model = None
scaler = None
knn = None
brands = []

# Load and preprocess data
def load_data():
    global df, model, scaler, knn, brands
    
    try:
        # Load the dataset
        df = pd.read_csv("mobile.csv")
        
        # Basic cleaning
        df = df.rename(columns={
            'brand_name': 'brand',
            'ram_capacity': 'ram',
            'internal_memory': 'storage',
            'battery_capacity': 'battery',
            'num_rear_cameras': 'rear_cameras',
            'refresh_rate': 'refresh_rate',
            'screen_size': 'screen_size',
            'processor_brand': 'processor'
        })
        
        # Convert storage to GB
        def convert_to_gb(x):
            if pd.isna(x):
                return 0
            if isinstance(x, str):
                if 'TB' in x.upper():
                    return float(x.upper().replace('TB', '').strip()) * 1024
                if 'GB' in x.upper():
                    return float(x.upper().replace('GB', '').strip())
            return float(x) if pd.notna(x) else 0
        
        df['storage'] = df['storage'].apply(convert_to_gb)
        
        # Fill missing values
        numeric_features = ['ram', 'storage', 'battery', 'screen_size', 'refresh_rate', 'rear_cameras']
        for feature in numeric_features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            df[feature] = df[feature].fillna(df[feature].median())
        
        # Get unique brands
        brands = sorted(df['brand'].str.upper().unique().tolist())
        
        # Features for prediction
        features = ['ram', 'storage', 'battery', 'screen_size', 'refresh_rate', 'rear_cameras']
        
        # Prepare data for KNN
        df_knn = df[features + ['price']].dropna()
        X_knn = df_knn[features]
        y_knn = df_knn['price']
        
        # Scale features for KNN
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_knn)
        
        # Train KNN model
        knn = NearestNeighbors(n_neighbors=5, metric='euclidean')
        knn.fit(X_scaled)
        
        # Train Linear Regression model
        X = df_knn[features]
        y = y_knn
        
        # Create preprocessing pipeline
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, features)
            ])
        
        # Create and train the model
        model = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', LinearRegression())
        ])
        
        model.fit(X, y)
        
        print("Data loaded and models trained successfully!")
        return df, model, scaler, knn, brands
        
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None, []

# Initialize the application
def initialize():
    global df, model, scaler, knn, brands
    df, model, scaler, knn, brands = load_data()

# Initialize data at startup
initialize()

# Routes
@app.route('/')
def home():
    return render_template('index.html', brands=brands)

@app.route('/predict', methods=['POST'])
def predict():
    global df, model, scaler, knn
    
    # Check if data is loaded
    if df is None or model is None or scaler is None or knn is None:
        try:
            # Try to reload data if not loaded
            load_data()
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Failed to load data: {str(e)}'
            }), 500
            
    try:
        # Get form data with validation
        try:
            ram = float(request.form.get('ram', 0))
            storage = float(request.form.get('storage', 0))
            battery = float(request.form.get('battery', 0))
            screen_size = float(request.form.get('screen_size', 0))
            refresh_rate = float(request.form.get('refresh_rate', 60))
            rear_cameras = int(request.form.get('rear_cameras', 2))
        except (ValueError, TypeError) as e:
            return jsonify({
                'status': 'error',
                'message': f'Invalid input data: {str(e)}'
            }), 400
        
        # Create a DataFrame with the input data
        import pandas as pd
        input_data = pd.DataFrame({
            'ram': [ram],
            'storage': [storage],
            'battery': [battery],
            'screen_size': [screen_size],
            'refresh_rate': [refresh_rate],
            'rear_cameras': [rear_cameras]
        })
        
        # Make prediction
        try:
            predicted_price = model.predict(input_data)[0]
        except Exception as e:
            return jsonify({
                'status': 'error',
                'message': f'Prediction failed: {str(e)}',
                'details': str(e.__class__.__name__)
            }), 500
        
        # Get similar phones
        similar_phones = []
        if knn is not None and scaler is not None and df is not None:
            try:
                # Prepare features for KNN
                features = ['ram', 'storage', 'battery', 'screen_size', 'refresh_rate', 'rear_cameras']
                X_knn = df[features].copy()
                
                # Scale the input features
                scaled_input = scaler.transform(input_data[features])
                
                # Get nearest neighbors
                distances, indices = knn.kneighbors(scaled_input, n_neighbors=min(5, len(df)))
                
                for i, idx in enumerate(indices[0]):
                    try:
                        phone = df.iloc[idx].to_dict()
                        phone['distance'] = float(distances[0][i])
                        similar_phones.append({
                            'brand': str(phone.get('brand', '')),
                            'model': str(phone.get('model', '')),
                            'price': float(phone.get('price', 0)),
                            'ram': float(phone.get('ram', 0)),
                            'storage': float(phone.get('storage', 0)),
                            'battery': float(phone.get('battery', 0)),
                            'screen_size': float(phone.get('screen_size', 0)),
                            'similarity_score': round((1 / (1 + float(phone.get('distance', 1)))) * 100, 1)
                        })
                    except Exception as e:
                        continue  # Skip this phone if there's an error
            except Exception as e:
                # Log the error but continue without similar phones
                print(f"Error finding similar phones: {str(e)}")
                pass
        
        return jsonify({
            'status': 'success',
            'predicted_price': round(predicted_price, 2),
            'similar_phones': similar_phones
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An unexpected error occurred: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
