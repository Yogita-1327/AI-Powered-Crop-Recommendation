#!/usr/bin/env python3
"""
Crop Advisory System - Main Implementation
Data-Driven Crop Recommendation using Random Forest and Regression
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def create_crop_dataset():
    """Create a comprehensive crop recommendation dataset"""
    np.random.seed(42)
    
    # Define crops with their optimal conditions
    crop_conditions = {
        'rice': {'N': (80, 120), 'P': (40, 60), 'K': (40, 60), 'temp': (20, 35), 'humidity': (80, 95), 'ph': (5.5, 7.0), 'rainfall': (150, 300)},
        'maize': {'N': (80, 120), 'P': (40, 80), 'K': (40, 80), 'temp': (18, 27), 'humidity': (60, 70), 'ph': (5.8, 8.0), 'rainfall': (50, 100)},
        'chickpea': {'N': (40, 70), 'P': (60, 80), 'K': (80, 120), 'temp': (20, 25), 'humidity': (10, 40), 'ph': (6.0, 7.5), 'rainfall': (30, 100)},
        'kidneybeans': {'N': (20, 40), 'P': (60, 80), 'K': (50, 70), 'temp': (15, 25), 'humidity': (18, 65), 'ph': (5.5, 7.0), 'rainfall': (60, 120)},
        'pigeonpeas': {'N': (20, 40), 'P': (60, 80), 'K': (60, 80), 'temp': (18, 29), 'humidity': (30, 65), 'ph': (4.5, 8.2), 'rainfall': (60, 180)},
        'cotton': {'N': (120, 160), 'P': (40, 60), 'K': (40, 60), 'temp': (21, 30), 'humidity': (50, 80), 'ph': (5.8, 8.0), 'rainfall': (50, 100)},
        'banana': {'N': (100, 120), 'P': (75, 85), 'K': (50, 60), 'temp': (26, 30), 'humidity': (75, 85), 'ph': (5.5, 7.0), 'rainfall': (100, 180)},
        'watermelon': {'N': (100, 120), 'P': (80, 120), 'K': (40, 50), 'temp': (24, 27), 'humidity': (80, 90), 'ph': (6.0, 7.0), 'rainfall': (40, 50)},
        'grapes': {'N': (20, 40), 'P': (125, 135), 'K': (200, 250), 'temp': (8, 22), 'humidity': (80, 90), 'ph': (5.5, 7.0), 'rainfall': (50, 125)},
        'apple': {'N': (20, 30), 'P': (125, 135), 'K': (200, 250), 'temp': (8, 22), 'humidity': (80, 90), 'ph': (5.5, 7.0), 'rainfall': (50, 125)}
    }
    
    data = []
    crops = list(crop_conditions.keys())
    
    # Generate 200 samples per crop
    for crop in crops:
        conditions = crop_conditions[crop]
        for _ in range(200):
            # Generate values within optimal ranges with some noise
            N = np.random.uniform(conditions['N'][0], conditions['N'][1]) + np.random.normal(0, 5)
            P = np.random.uniform(conditions['P'][0], conditions['P'][1]) + np.random.normal(0, 3)
            K = np.random.uniform(conditions['K'][0], conditions['K'][1]) + np.random.normal(0, 3)
            temperature = np.random.uniform(conditions['temp'][0], conditions['temp'][1]) + np.random.normal(0, 2)
            humidity = np.random.uniform(conditions['humidity'][0], conditions['humidity'][1]) + np.random.normal(0, 5)
            ph = np.random.uniform(conditions['ph'][0], conditions['ph'][1]) + np.random.normal(0, 0.2)
            rainfall = np.random.uniform(conditions['rainfall'][0], conditions['rainfall'][1]) + np.random.normal(0, 10)
            
            # Ensure values are within reasonable bounds
            N, P, K = max(0, N), max(0, P), max(0, K)
            temperature = max(0, temperature)
            humidity = max(0, min(100, humidity))
            ph = max(0, min(14, ph))
            rainfall = max(0, rainfall)
            
            data.append([N, P, K, temperature, humidity, ph, rainfall, crop])
    
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    return df

def analyze_dataset(df):
    """Perform basic data analysis"""
    print("=== CROP ADVISORY SYSTEM - DATASET ANALYSIS ===")
    print(f"Dataset Shape: {df.shape}")
    print(f"Number of Crops: {df['label'].nunique()}")
    print(f"Crops: {', '.join(sorted(df['label'].unique()))}")
    print("\nDataset Info:")
    print(df.info())
    print("\nStatistical Summary:")
    print(df.describe())
    print("\nCrop Distribution:")
    print(df['label'].value_counts())
    
    return df

def train_models(df):
    """Train Random Forest and Logistic Regression models"""
    print("\n=== MODEL TRAINING ===")
    
    # Prepare features and target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    y = df['label']
    
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest Classifier...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    
    # Train Logistic Regression
    print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_pred = lr_model.predict(X_test_scaled)
    lr_accuracy = accuracy_score(y_test, lr_pred)
    
    print(f"\nModel Performance:")
    print(f"Random Forest Accuracy: {rf_accuracy:.4f}")
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    
    # Feature importance for Random Forest
    feature_names = ['N', 'P', 'K', 'Temperature', 'Humidity', 'pH', 'Rainfall']
    feature_importance = rf_model.feature_importances_
    
    print("\nFeature Importance (Random Forest):")
    for name, importance in zip(feature_names, feature_importance):
        print(f"{name}: {importance:.4f}")
    
    return {
        'rf_model': rf_model, 
        'lr_model': lr_model, 
        'scaler': scaler,
        'X_test': X_test,
        'X_test_scaled': X_test_scaled,
        'y_test': y_test,
        'rf_pred': rf_pred,
        'lr_pred': lr_pred,
        'feature_names': feature_names,
        'feature_importance': feature_importance
    }

def create_visualizations(df, models_data):
    """Create visualizations for PowerBI integration"""
    print("\n=== CREATING VISUALIZATIONS ===")
    
    # Create visualization data for PowerBI
    viz_data = []
    
    # 1. Crop distribution
    crop_counts = df['label'].value_counts()
    for crop, count in crop_counts.items():
        viz_data.append({
            'visualization_type': 'crop_distribution',
            'crop': crop,
            'count': count,
            'percentage': (count/len(df))*100
        })
    
    # 2. Feature importance
    for feature, importance in zip(models_data['feature_names'], models_data['feature_importance']):
        viz_data.append({
            'visualization_type': 'feature_importance',
            'feature': feature,
            'importance': importance
        })
    
    # 3. Model performance
    viz_data.append({
        'visualization_type': 'model_performance',
        'model': 'Random Forest',
        'accuracy': accuracy_score(models_data['y_test'], models_data['rf_pred'])
    })
    viz_data.append({
        'visualization_type': 'model_performance',
        'model': 'Logistic Regression', 
        'accuracy': accuracy_score(models_data['y_test'], models_data['lr_pred'])
    })
    
    # 4. Average conditions per crop
    crop_avg_conditions = df.groupby('label')[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].mean()
    for crop in crop_avg_conditions.index:
        for feature in crop_avg_conditions.columns:
            viz_data.append({
                'visualization_type': 'crop_conditions',
                'crop': crop,
                'parameter': feature,
                'average_value': crop_avg_conditions.loc[crop, feature]
            })
    
    # Save visualization data for PowerBI
    viz_df = pd.DataFrame(viz_data)
    viz_df.to_csv('visualizations/powerbi_data.csv', index=False)
    print("✓ PowerBI visualization data saved to 'visualizations/powerbi_data.csv'")
    
    return viz_df

def crop_recommendation_system(models_data):
    """Interactive crop recommendation system"""
    print("\n=== CROP RECOMMENDATION SYSTEM ===")
    print("Enter soil and environmental parameters for crop recommendation:")
    
    try:
        # Get user input
        N = float(input("Nitrogen (N): "))
        P = float(input("Phosphorus (P): "))
        K = float(input("Potassium (K): "))
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        ph = float(input("pH level: "))
        rainfall = float(input("Rainfall (mm): "))
        
        # Make predictions
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        input_data_scaled = models_data['scaler'].transform(input_data)
        
        rf_prediction = models_data['rf_model'].predict(input_data)[0]
        rf_probability = max(models_data['rf_model'].predict_proba(input_data)[0])
        
        lr_prediction = models_data['lr_model'].predict(input_data_scaled)[0]
        lr_probability = max(models_data['lr_model'].predict_proba(input_data_scaled)[0])
        
        print(f"\n--- CROP RECOMMENDATIONS ---")
        print(f"Random Forest Recommendation: {rf_prediction} (Confidence: {rf_probability:.2f})")
        print(f"Logistic Regression Recommendation: {lr_prediction} (Confidence: {lr_probability:.2f})")
        
        # Get top 3 recommendations from Random Forest
        rf_proba = models_data['rf_model'].predict_proba(input_data)[0]
        top_3_indices = rf_proba.argsort()[-3:][::-1]
        classes = models_data['rf_model'].classes_
        
        print(f"\nTop 3 Crop Recommendations:")
        for i, idx in enumerate(top_3_indices, 1):
            print(f"{i}. {classes[idx]} (Probability: {rf_proba[idx]:.3f})")
        
    except ValueError:
        print("Please enter valid numeric values.")
    except KeyboardInterrupt:
        print("\nExiting recommendation system.")

def main():
    """Main function to run the crop advisory system"""
    print("CROP ADVISORY SYSTEM")
    print("=" * 50)
    
    # Create dataset
    print("Creating crop recommendation dataset...")
    df = create_crop_dataset()
    df.to_csv('data/crop_recommendation.csv', index=False)
    print("✓ Dataset saved to 'data/crop_recommendation.csv'")
    
    # Analyze dataset
    df = analyze_dataset(df)
    
    # Train models
    models_data = train_models(df)
    
    # Create visualizations
    viz_df = create_visualizations(df, models_data)
    
    # Interactive recommendation system
    while True:
        try:
            continue_rec = input("\nWould you like to get crop recommendations? (y/n): ")
            if continue_rec.lower() != 'y':
                break
            crop_recommendation_system(models_data)
        except KeyboardInterrupt:
            break
    
    print("\n✓ Crop Advisory System completed successfully!")
    print("Files generated:")
    print("- data/crop_recommendation.csv (main dataset)")
    print("- visualizations/powerbi_data.csv (PowerBI visualization data)")

if __name__ == "__main__":
    main()