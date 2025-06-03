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
        # Load the dataset with proper encoding
        df = pd.read_csv("mobile.csv", encoding='utf-8')
        
        # Create a copy of the dataframe to avoid SettingWithCopyWarning
        df = df.copy()
        
        # Basic cleaning and renaming columns to match our expected format
        df = df.rename(columns={
            'brand_name': 'brand',
            'ram_capacity': 'ram',
            'internal_memory': 'storage',
            'battery_capacity': 'battery',
            'num_rear_cameras': 'rear_cameras',
            'refresh_rate': 'refresh_rate',
            'screen_size': 'screen_size',
            'processor_brand': 'processor',
            'price': 'price',
            'model': 'model',
            'avg_rating': 'rating',
            'primary_camera_rear': 'rear_camera',
            'primary_camera_front': 'front_camera',
            'resolution_height': 'resolution_height',
            'resolution_width': 'resolution_width',
            '5G_or_not': 'has_5g',
            'fast_charging_available': 'fast_charging_available',
            'fast_charging': 'fast_charging_watts',
            'extended_memory_available': 'has_memory_card'
        })
        
        # Convert storage to GB - handle different formats
        def convert_to_gb(x):
            try:
                if pd.isna(x) or x == '':
                    return 0
                if isinstance(x, str):
                    x = x.strip().upper()
                    if 'TB' in x:
                        return float(x.replace('TB', '').strip()) * 1024
                    if 'GB' in x:
                        return float(x.replace('GB', '').strip())
                    if 'MB' in x:
                        return float(x.replace('MB', '').strip()) / 1024
                return float(x) if pd.notna(x) and str(x).strip() != '' else 0
            except Exception as e:
                print(f"Error converting value '{x}' to GB: {e}")
                return 0
        
        # Apply conversion to storage column
        df['storage'] = df['storage'].apply(convert_to_gb).astype(float)
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['ram', 'battery', 'screen_size', 'refresh_rate', 'rear_cameras',
                         'price', 'rating', 'rear_camera', 'front_camera',
                         'resolution_height', 'resolution_width', 'fast_charging_watts']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Fill missing values
        numeric_features = ['ram', 'storage', 'battery', 'screen_size', 'refresh_rate', 'rear_cameras']
        for feature in numeric_features:
            if feature in df.columns:
                median_value = df[feature].median()
                df[feature] = df[feature].fillna(median_value).astype(float)
        
        # Fill missing values
        numeric_features = ['ram', 'storage', 'battery', 'screen_size', 'refresh_rate', 'rear_cameras']
        for feature in numeric_features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            median_value = df[feature].median()
            df[feature] = df[feature].fillna(median_value).astype(float)
        
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
                'message': f'Invalid input values: {str(e)}. Please check your input and try again.'
            }), 400
            
        # Prepare input for prediction with all required features
        input_data = pd.DataFrame({
            'ram': [ram],
            'storage': [storage],
            'battery': [battery],
            'screen_size': [screen_size],
            'refresh_rate': [refresh_rate],
            'rear_cameras': [rear_cameras]
        })
        
        # Ensure all numeric columns are float
        for col in input_data.columns:
            input_data[col] = pd.to_numeric(input_data[col], errors='coerce')
        
        # Fill any remaining NaN values with column means from training data
        for col in input_data.columns:
            if input_data[col].isna().any() and col in df.columns:
                input_data[col] = input_data[col].fillna(df[col].median())
        
        # Predict price
        predicted_price = model.predict(input_data)[0]
        
        # Prepare features for KNN
        features = ['ram', 'storage', 'battery', 'screen_size', 'refresh_rate', 'rear_cameras']
        
        # Ensure all features exist in the dataframe
        available_features = [f for f in features if f in df.columns]
        X = df[available_features].values
        
        # Scale input for KNN using the same scaler
        try:
            scaled_input = scaler.transform(input_data[available_features])
        except Exception as e:
            print(f"Error scaling input: {e}")
            # If scaling fails, use original values
            scaled_input = input_data[available_features].values
        
        # Find nearest neighbors
        try:
            distances, indices = knn.kneighbors(scaled_input, n_neighbors=min(5, len(df)))
        except Exception as e:
            print(f"Error in KNN: {e}")
            # If KNN fails, return random phones as fallback
            indices = np.random.choice(len(df), min(5, len(df)), replace=False).reshape(1, -1)
            distances = np.ones_like(indices, dtype=float)
        
        # Prepare similar phones data with proper error handling
        similar_phones = []
        for i, idx in enumerate(indices[0]):
            try:
                phone = df.iloc[idx].to_dict()
                similar_phones.append({
                    'id': int(idx),
                    'brand': str(phone.get('brand', 'Unknown')),
                    'model': str(phone.get('model', 'Unknown')),
                    'price': float(phone.get('price', 0)),
                    'ram': float(phone.get('ram', 0)),
                    'storage': float(phone.get('storage', 0)),
                    'battery': float(phone.get('battery', 0)),
                    'screen_size': float(phone.get('screen_size', 0)),
                    'refresh_rate': float(phone.get('refresh_rate', 60)),
                    'rear_cameras': float(phone.get('rear_cameras', 2)),
                    'similarity_score': min(round((1 - (distances[0][i] / (distances[0].max() or 1))) * 100, 1), 100)
                })
            except Exception as e:
                print(f"Error processing phone {idx}: {e}")
                continue
        
        # Ensure we have at least one similar phone
        if not similar_phones:
            # Fallback to random phones if no similar phones found
            random_indices = np.random.choice(len(df), min(5, len(df)), replace=False)
            for idx in random_indices:
                phone = df.iloc[idx].to_dict()
                similar_phones.append({
                    'id': int(idx),
                    'brand': str(phone.get('brand', 'Unknown')),
                    'model': str(phone.get('model', 'Unknown')),
                    'price': float(phone.get('price', 0)),
                    'ram': float(phone.get('ram', 0)),
                    'storage': float(phone.get('storage', 0)),
                    'battery': float(phone.get('battery', 0)),
                    'screen_size': float(phone.get('screen_size', 0)),
                    'refresh_rate': float(phone.get('refresh_rate', 60)),
                    'rear_cameras': float(phone.get('rear_cameras', 2)),
                    'similarity_score': round(90 - (np.random.random() * 20), 1)  # Random score between 70-90
                })
        
        return jsonify({
            'status': 'success',
            'predicted_price': float(predicted_price),
            'similar_phones': similar_phones
        })
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': f'An unexpected error occurred: {str(e)}'
        })

@app.route('/phone/<int:phone_id>')
def phone_details(phone_id):
    global df, knn, scaler
    
    try:
        print(f"\n=== DEBUG: Starting phone_details for phone_id: {phone_id} ===", flush=True)
        print(f"Request URL: {request.url}", flush=True)
        print(f"Request method: {request.method}", flush=True)
        print(f"Request headers: {dict(request.headers)}", flush=True)
        
        # Check if dataframe is loaded
        if df is None:
            print("ERROR: DataFrame is None")
            return render_template('error.html', 
                               error_message='Phone database not loaded', 
                               status_code=500)
        if df.empty:
            print("ERROR: DataFrame is empty")
            return render_template('error.html', 
                               error_message='Phone database is empty', 
                               status_code=500)
        
        print(f"DataFrame shape: {df.shape}")
        print(f"DataFrame columns: {df.columns.tolist()}")
        
        # Check if the phone_id is valid
        if phone_id < 0 or phone_id >= len(df):
            print(f"ERROR: Invalid phone_id: {phone_id} (valid range: 0-{len(df)-1})")
            return render_template('error.html', 
                               error_message=f'Phone with ID {phone_id} not found. Please try a different phone.', 
                               status_code=404)
        
        # Get phone details
        try:
            phone = df.iloc[phone_id].to_dict()
            print(f"Retrieved phone data: {phone.get('brand', 'Unknown')} {phone.get('model', 'Unknown')}")
        except Exception as e:
            print(f"ERROR: Failed to get phone details: {str(e)}")
            return render_template('error.html',
                               error_message='Failed to retrieve phone details',
                               status_code=500)
        
        # Helper function to safely get values with formatting
        def get_value(value, prefix='', suffix='', default='N/A', is_percent=False, is_currency=False):
            if pd.isna(value) or value == '':
                return default
            try:
                if is_currency:
                    return f"₹{float(value):,.2f}"
                if is_percent:
                    return f"{float(value):.1f}%"
                if suffix == ' GB' and float(value) >= 1024:
                    return f"{float(value)/1024:.1f} TB"
                return f"{prefix}{value}{suffix}"
            except (ValueError, TypeError):
                return str(value) if value != '' else default
        
        # Get similar phones using KNN
        similar_phones = []
        if knn is not None and scaler is not None:
            try:
                # Prepare features for KNN
                features = ['ram', 'storage', 'battery', 'screen_size', 'refresh_rate', 'rear_cameras']
                available_features = [f for f in features if f in df.columns and f in phone and pd.notna(phone[f])]
                
                if available_features:
                    # Prepare input for KNN
                    input_data = pd.DataFrame([{f: phone[f] for f in available_features}])
                    
                    # Scale the input
                    scaled_input = scaler.transform(input_data)
                    
                    # Find nearest neighbors (excluding the phone itself)
                    distances, indices = knn.kneighbors(scaled_input, n_neighbors=min(6, len(df)))
                    
                    # Get similar phones (skip the first one as it's the phone itself)
                    for i, idx in enumerate(indices[0][1:], 1):
                        try:
                            similar_phone = df.iloc[idx].to_dict()
                            similar_phones.append({
                                'id': int(idx),
                                'brand': str(similar_phone.get('brand', 'Unknown')),
                                'model': str(similar_phone.get('model', 'Unknown')),
                                'price': float(similar_phone.get('price', 0)),
                                'rating': float(similar_phone.get('rating', 0)) if pd.notna(similar_phone.get('rating')) else None,
                                'similarity': 100 - (distances[0][i] * 10)  # Convert distance to similarity score (0-100)
                            })
                        except Exception as e:
                            print(f"Error processing similar phone {idx}: {e}")
                            continue
            except Exception as e:
                print(f"Error finding similar phones: {e}")
        
        # Prepare phone specifications with proper formatting and fallbacks
        specs = {
            'General': {
                'Brand': phone.get('brand', 'N/A').title() if pd.notna(phone.get('brand')) else 'N/A',
                'Model': str(phone.get('model', 'N/A')).title(),
                'Price': get_value(phone.get('price'), is_currency=True),
                'Rating': get_value(phone.get('rating'), suffix='/10'),
                'Operating System': str(phone.get('os', 'N/A')).capitalize(),
                '5G Support': 'Yes' if phone.get('has_5g') == 1 else 'No',
                'Fast Charging': 'Yes' if phone.get('fast_charging_available') == 1 else 'No',
                'Fast Charging Power': get_value(phone.get('fast_charging_watts'), suffix='W'),
                'Memory Card Support': 'Yes' if phone.get('has_memory_card') == 1 else 'No'
            },
            'Performance': {
                'Processor': str(phone.get('processor', 'N/A')).title(),
                'Number of Cores': get_value(phone.get('num_cores')),
                'Processor Speed': get_value(phone.get('processor_speed'), suffix=' GHz'),
                'RAM': get_value(phone.get('ram'), suffix=' GB'),
                'Storage': get_value(phone.get('storage'), suffix=' GB')
            },
            'Display': {
                'Screen Size': get_value(phone.get('screen_size'), suffix='"'),
                'Refresh Rate': get_value(phone.get('refresh_rate'), suffix=' Hz'),
                'Resolution': f"{int(phone.get('resolution_width', 0))} × {int(phone.get('resolution_height', 0))}" 
                             if pd.notna(phone.get('resolution_width')) and pd.notna(phone.get('resolution_height')) 
                             and int(phone.get('resolution_width', 0)) > 0 and int(phone.get('resolution_height', 0)) > 0
                             else 'N/A',
                'Aspect Ratio': 'N/A'  # Can be calculated from resolution if needed
            },
            'Camera': {
                'Rear Cameras': get_value(phone.get('rear_cameras')),
                'Primary Rear Camera': get_value(phone.get('rear_camera'), suffix=' MP'),
                'Front Camera': get_value(phone.get('front_camera'), suffix=' MP'),
                'Camera Features': 'N/A'  # Can be enhanced with more details if available
            },
            'Battery': {
                'Capacity': get_value(phone.get('battery'), suffix=' mAh'),
                'Fast Charging': 'Yes' if phone.get('fast_charging_available') == 1 else 'No',
                'Fast Charging Power': get_value(phone.get('fast_charging_watts'), suffix='W')
            },
            'Connectivity': {
                '5G': 'Yes' if phone.get('has_5g') == 1 else 'No',
                'Wi-Fi': 'Yes',  # Assuming all phones have Wi-Fi
                'Bluetooth': 'Yes',  # Assuming all phones have Bluetooth
                'NFC': 'N/A',  # Not in current dataset
                'USB Type': 'Type-C'  # Assuming modern phones use Type-C
            }
        }
        
        # Add key features from the phone
        key_features = []
        if phone.get('has_5g') == 1:
            key_features.append('5G Support')
        if phone.get('fast_charging_available') == 1:
            key_features.append('Fast Charging')
        if phone.get('has_memory_card') == 1:
            key_features.append('Expandable Storage')
        if phone.get('refresh_rate', 0) >= 90:
            key_features.append('High Refresh Rate Display')
        if phone.get('ram', 0) >= 8:
            key_features.append('High RAM')
        if phone.get('storage', 0) >= 256:
            key_features.append('High Storage')
            
        phone['key_features'] = key_features
        
        # Clean up any None values in specs
        for section in specs:
            specs[section] = {k: v if v is not None else 'N/A' for k, v in specs[section].items()}
        
        # Get phone name for the page title
        phone_name = f"{phone.get('brand', '')} {phone.get('model', '')}".strip()
        
        return render_template('phone_details.html', 
                             phone=phone, 
                             specs=specs,
                             similar_phones=similar_phones[:5],  # Limit to 5 similar phones
                             title=f"{phone_name} - Specifications" if phone_name else "Phone Specifications")
        
    except Exception as e:
        import traceback
        import sys
        
        # Get the error traceback as a string
        error_trace = traceback.format_exc()
        error_msg = f"Error in phone_details: {str(e)}"
        
        # Print error information in a way that handles Unicode characters
        try:
            # Try to print with UTF-8 encoding
            print(f"\n=== CRITICAL ERROR ===\n{error_msg}\n{error_trace}", file=sys.stderr)
        except UnicodeEncodeError:
            # If UTF-8 fails, print a simplified error message
            print("\n=== CRITICAL ERROR ===", file=sys.stderr)
            print("An error occurred while processing the request.", file=sys.stderr)
            print(f"Error type: {type(e).__name__}", file=sys.stderr)
        
        # Log additional debug information
        debug_info = {
            'phone_id': phone_id,
            'df_loaded': df is not None,
            'df_columns': list(df.columns) if df is not None else [],
            'df_length': len(df) if df is not None else 0,
            'error': str(e).encode('ascii', 'replace').decode('ascii')  # Convert to ASCII with replacement
        }
        
        try:
            print(f"Debug Info: {debug_info}", file=sys.stderr)
        except UnicodeEncodeError:
            print("Debug Info: [Unable to display due to encoding issues]", file=sys.stderr)
        
        # Return a simple error response
        error_message = 'An unexpected error occurred while loading phone details. Please try again later.'
        return render_template('error.html', 
                           error_message=error_message, 
                           status_code=500,
                           debug_info=debug_info if app.debug else None)

# Create templates directory if it doesn't exist
os.makedirs('templates', exist_ok=True)

# Add a test route to verify the app is running
@app.route('/test')
def test_route():
    return jsonify({
        'status': 'success',
        'message': 'API is working!',
        'data_loaded': df is not None,
        'num_phones': len(df) if df is not None else 0
    })

# Add a simple test route
@app.route('/test')
def test():
    return jsonify({
        'status': 'success',
        'message': 'API is working!',
        'data_loaded': df is not None,
        'num_phones': len(df) if df is not None else 0
    })

# Run the app with debug mode
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))  # Changed default port to 8080
    app.run(debug=True, host='0.0.0.0', port=port)
