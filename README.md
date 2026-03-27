# Heart Disease Prediction Analysis

This project utilizes the Cleveland Heart Disease dataset to build and evaluate multiple machine learning models. The primary goal is to classify the presence of heart disease (condition) based on clinical features such as age, cholesterol levels, and maximum heart rate.

## Project Structure

The analysis follows a standard data science workflow:

### 1. Environment Setup and Data Loading
* **Environment:** Configured for Google Colab with Google Drive integration.
* **Dependencies:** Includes `pandas`, `scikit-learn`, `seaborn`, `xgboost`, and `statsmodels`.
* **Data:** Uses `heart_cleveland_upload.csv`, consisting of 13 features and 1 target variable (`condition`).

### 2. Exploratory Data Analysis (EDA)
* **Visualizations:** Boxplots, violin plots, and stripplots are used to examine the distribution of age and other continuous variables against the target condition.
* **Correlation Analysis:** A heatmap is generated to identify relationships between features and the target variable.

### 3. Data Preprocessing
* **Encoding:** Categorical variables are handled via `pd.get_dummies`.
* **Splitting:** The data is divided into training and testing sets using a $70/30$ split.
* **Scaling:** Features are normalized using `StandardScaler` to ensure optimal performance for distance-based and gradient-based models.

### 4. Model Implementation and Evaluation
The project implements and compares four distinct classification algorithms:

* **Logistic Regression:** Includes odds ratio analysis and a classification report.
* **K-Nearest Neighbors (K-NN):** Features a search for the optimal $k$ value by plotting AUC scores across 30 neighbors.
* **Random Forest:** Evaluates performance and identifies the most influential features via a Feature Importance plot.
* **XGBoost:** A gradient-boosted decision tree approach for high-performance classification.

### 5. Comparative Results
The final section of the notebook merges the ROC curves for all models into a single visualization to compare the Area Under the Curve (AUC) across:
* Logistic Regression
* K-NN
* Random Forest
* XGBoost

## Usage
To run this analysis, ensure all dependencies are installed:
`pip install sktime dmba prince plotly yellowbrick xgboost`

Run the script in a Jupyter environment or a Python interpreter that supports Matplotlib rendering.
