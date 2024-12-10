# House-Price-Prediction-Enhancing-Model-Accuracy
This repository contains a comprehensive machine learning pipeline for predicting house prices using linear regression and ridge regression techniques. 

The project **aims** to explore and preprocess a housing dataset, analyze relationships between variables, and build robust predictive models.

## Overview
The project consists of the following key steps:
1. **Data Import and Exploration:** Loading the dataset, understanding its structure, and identifying initial patterns.
2. **Data Visualization:** Using Seaborn and Matplotlib for exploratory data analysis to uncover relationships between features.
3. **Data Cleaning:** Handling missing values and treating outliers to ensure data quality.
4. **Feature Engineering:** Creating new features and removing irrelevant ones for better model performance.
5. **Model Building:** Developing and evaluating multiple regression models using `LinearRegression` and `Ridge`.


---

## Analysis Observations
1. **Outliers:**
   - High values in `n_hot_rooms` were capped to reduce their impact.
   - Low values in `rainfall` were adjusted to remove anomalies.

2. **Missing Values:**
   - The column `n_hos_beds` had missing values filled with the mean.

3. **Feature Transformations:**
   - Log transformation was applied to `crime_rate` to normalize its distribution.
   - Average distance (`avg_dist`) was computed as a new feature.

4. **Dummy Variables:**
   - Categorical variables were encoded to prepare the dataset for regression.

5. **Correlation Analysis:**
   - A heatmap highlighted the relationships between variables, aiding in feature selection.

---

## Insights
- **Room Number vs. Price:** A positive correlation indicates that more rooms generally lead to higher prices.
- **Proximity Metrics:** Variables representing distances to amenities were reduced to a single averaged metric, simplifying the dataset without losing information.
- **Categorical Variables:** Features such as `airport` and `waterbody` were found to significantly influence price.

---

## Recommendations
1. Enhance feature selection by including interaction terms or polynomial features for non-linear relationships.
2. Increase the R² score by using ensemble methods like Gradient Boosting or Random Forest.
3. Use cross-validation extensively to minimize overfitting.
4. Deploy the model using a user-friendly interface for real-world applicability.

---

## Links
- [House Price Prediction Notebook](https://github.com/EngrIBGIT/House-Price-Prediction-Enhancing-Model-Accuracy/blob/main/House_Price_mod.ipynb) 
- [Deployment Link](https://house-price-prediction-enhancing-model-2vdp.onrender.com)

---

### Repository Structure
|-- data/ | |-- House_Price.csv |-- notebooks/ | |-- House_Price_Analysis.ipynb |-- scripts/ | |-- data_preprocessing.py | |-- model_training.py |-- README.md



---

Contributions are welcome! Please raise an issue or submit a pull request for improvements.
