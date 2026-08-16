# 🚗 CarPrice AI — Used Car Price Prediction

A practical Machine Learning regression project that predicts the estimated price of a used car in Egyptian Pounds (EGP) based on its specifications.

## 🎯 Project Objective

The goal of **CarPrice AI** is to build a Machine Learning model that learns the relationship between used-car characteristics and their prices, then provides an estimated price through a simple GUI.

## 📥 Input Features

The model uses:

- Brand
- Model
- Kilometers
- Year
- Fuel Type
- Transmission Type
- Engine Capacity (CC)
- Body Type

### 🎯 Target

**Price_EGP** — the estimated used-car price in Egyptian Pounds.

## 🧹 Data Preprocessing

The project handles missing and inconsistent data using the following rules:

- If **Brand** is missing → drop the entire row.
- If **Model** is missing → drop the entire row.
- If **Fuel Type** is missing → replace it with the Mode.
- If **Body Type** is missing → replace it with the Mode.
- Missing numerical values → handled using **KNN Imputation**.

Prices such as:

```text
EGP 140,000
```

are converted into:

```text
140000
```

Categorical features are transformed using **One-Hot Encoding**.

## 🤖 Machine Learning Model

The project uses a:

**Random Forest Regressor**

Random Forest is suitable for this problem because car prices can have non-linear relationships with features such as mileage, year, brand, model, and engine capacity.

## 🧠 KNN Imputation

For missing numerical values, the project uses:

```python
KNNImputer(n_neighbors=5)
```

The imputer estimates missing values based on similar observations in the dataset.

## 📊 Evaluation Metrics

The model is evaluated using:

- **MAE (Mean Absolute Error)** — average absolute prediction error.
- **MSE (Mean Squared Error)** — average squared prediction error.
- **RMSE (Root Mean Squared Error)** — error in the same unit as the target.
- **R² (R-Squared)** — how well the model explains the variation in car prices.

## 🖥️ GUI

The application provides a graphical interface where the user enters the car information:

```text
Brand
Model
Kilometers
Year
Fuel Type
Transmission Type
Engine Capacity
Body Type
```

After clicking **PREDICT PRICE**, the application displays the estimated price in EGP.

Example:

```text
Estimated Price

450,000 EGP
```

## 📁 Project Structure

```text
CarPrice-AI/
│
├── used_car_price_app.py
├── dataset.csv
├── requirements.txt
└── README.md
```

The dataset is loaded directly using Pandas:

```python
df = pd.read_csv(CSV_PATH)
```

## ⚙️ Installation

Install the required libraries:

```bash
pip install pandas numpy scikit-learn
```

## ▶️ Running the Project

Set the path of your dataset:

```python
CSV_PATH = r"D:\path\to\your\dataset.csv"
```

Then run:

```bash
python used_car_price_app.py
```

The application will:

1. Load the dataset.
2. Clean the data.
3. Handle missing values.
4. Split the data into training and testing sets.
5. Apply preprocessing.
6. Train the Random Forest model.
7. Evaluate the model.
8. Open the GUI.
9. Predict the car price based on the entered features.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Tkinter
- Random Forest Regressor
- KNN Imputation
- One-Hot Encoding

## 📈 Machine Learning Pipeline

```text
Raw Dataset
     ↓
Data Cleaning
     ↓
Drop Missing Brand / Model
     ↓
Mode Imputation
     ↓
KNN Imputation
     ↓
Train / Test Split
     ↓
One-Hot Encoding
     ↓
Random Forest Regressor
     ↓
Evaluation
     ↓
GUI
     ↓
Predicted Car Price
```



