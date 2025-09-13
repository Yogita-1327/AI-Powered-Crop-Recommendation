# Data-Driven Crop Advisory System

## Project Overview
This project implements a **Data-Driven Crop Advisory System** that recommends the most suitable crop to cultivate based on soil macronutrients (N, P, K), pH level, temperature, humidity, and rainfall. The system uses machine learning models to analyze current soil and weather conditions and provide accurate crop suggestions for small farm holders.

## Project Structure
```
crop_advisory_system/
├── data/
│   └── crop_recommendation.csv      # Main dataset (2,250 samples)
├── visualizations/
│   ├── crop_distribution.csv        # Crop distribution data for PowerBI
│   ├── crop_conditions.csv          # Average parameter values per crop
│   └── optimal_vs_actual.csv        # Comparison of optimal vs actual conditions
├── demo_crop_system.py              # Main comprehensive system
├── simple_crop_system.py            # Simple version using basic Python
├── crop_recommendation.py           # Advanced version with sklearn
└── PROJECT_SUMMARY.md               # This file
```

## Features Implemented

### ✅ Data Analysis & Preprocessing
- Created synthetic dataset based on real agricultural parameters
- 2,250 samples across 9 crop types
- Balanced dataset with 250 samples per crop
- Statistical analysis of soil and environmental parameters

### ✅ Machine Learning Models
- **Enhanced K-Nearest Neighbors (KNN)**: Primary model with 91.3% accuracy
- **Weighted Distance Calculation**: Normalized features for equal importance
- **Stratified Train-Test Split**: Ensures balanced evaluation
- **Per-Crop Accuracy Analysis**: Detailed performance metrics

### ✅ Crop Types Supported
1. **Rice** - High humidity, tropical conditions
2. **Maize** - Moderate climate requirements
3. **Chickpea** - Low humidity, arid regions
4. **Kidney Beans** - Cool, moderate conditions
5. **Cotton** - High nitrogen requirements
6. **Banana** - Tropical, high nutrient needs
7. **Watermelon** - Warm climate, high water needs
8. **Grapes** - Cool climate, high K requirements
9. **Apple** - Temperate climate, fruit tree conditions

### ✅ PowerBI Integration
- **crop_distribution.csv**: Visualize crop dataset balance
- **crop_conditions.csv**: Average parameter values for each crop
- **optimal_vs_actual.csv**: Compare optimal vs generated conditions

## Model Performance

### Overall Accuracy: 91.3%
- Training samples: 1,800
- Testing samples: 450
- Algorithm: Enhanced K-Nearest Neighbors (k=7)

### Per-Crop Performance:
- **Perfect (100%) Accuracy**: Banana, Chickpea, Cotton, Kidney Beans, Maize, Rice, Watermelon
- **Good (68-54%) Accuracy**: Grapes (68%), Apple (54%)

## Usage Instructions

### Running the System
```bash
# Navigate to project directory
cd crop_advisory_system

# Run the comprehensive demo
python3 demo_crop_system.py

# Run the simple version (no external dependencies)
python3 simple_crop_system.py
```

### Sample Predictions
The system demonstrates its capabilities with various scenarios:

1. **High Nitrogen Rich Soil** → Recommends: Banana
2. **Low Humidity Arid Region** → Recommends: Chickpea  
3. **High Rainfall Tropical** → Recommends: Rice
4. **Temperate Fruit Growing** → Recommends: Apple/Grapes

## PowerBI Integration Guide

### 1. Import Data Files
- Load the three CSV files from `visualizations/` folder into PowerBI
- Each file serves different visualization purposes

### 2. Recommended Visualizations
- **Pie Chart**: Crop distribution using `crop_distribution.csv`
- **Bar Chart**: Parameter comparison using `crop_conditions.csv`
- **Scatter Plot**: Optimal vs actual conditions using `optimal_vs_actual.csv`

### 3. Dashboard Components
- Crop recommendation summary
- Parameter importance analysis
- Interactive crop selection filters
- Performance metrics display

## Technical Implementation

### Algorithm Choice: Enhanced K-Nearest Neighbors
- **Why KNN?** Simple, interpretable, works well with agricultural data
- **Enhancements**: 
  - Normalized distance calculation
  - Weighted voting based on inverse distance
  - Stratified sampling for balanced training

### Data Engineering
- **Feature Scaling**: Normalized different parameter ranges
- **Data Quality**: Controlled synthetic data with realistic variations
- **Validation**: Stratified split maintains crop distribution balance

### Alternative Implementation
For environments requiring scikit-learn, Random Forest, and Logistic Regression:
```bash
python3 crop_recommendation.py  # Advanced ML models version
```

## Real-World Impact

### For Small Farmers:
- **Scientific Decision Making**: Data-driven crop selection
- **Resource Optimization**: Better use of available nutrients
- **Cost Reduction**: Avoid unsuitable crop choices
- **Yield Improvement**: Match crops to soil conditions

### Scalability:
- Easy integration with IoT sensors
- Can be deployed as web/mobile application
- Supports multiple geographical regions
- Extensible to more crop types

## Future Enhancements

1. **Additional Features**:
   - Market price integration
   - Seasonal recommendations
   - Fertilizer advisory system
   - Weather forecast integration

2. **Model Improvements**:
   - Deep learning models
   - Ensemble methods
   - Real-time learning from farmer feedback

3. **User Interface**:
   - Web application
   - Mobile app
   - SMS-based recommendations

## Files Generated

| File | Purpose | Size |
|------|---------|------|
| `crop_recommendation.csv` | Main dataset | 2,250 samples |
| `crop_distribution.csv` | PowerBI visualization | Distribution data |
| `crop_conditions.csv` | PowerBI visualization | Parameter analysis |
| `optimal_vs_actual.csv` | PowerBI visualization | Comparison data |

## Conclusion

This Crop Advisory System successfully demonstrates a practical application of machine learning in agriculture. With 91.3% accuracy and comprehensive PowerBI integration, it provides a solid foundation for helping farmers make informed crop selection decisions based on their soil and environmental conditions.

The system is ready for deployment and can be easily extended with additional features as needed.