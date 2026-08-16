
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


CSV_PATH = r"D:\NTI Summer Training 2026\GP\car_ads_details_kaggle.csv"


NUMERIC_FEATURES = ["Kilometers", "Year", "Engine Capacity (CC)"]
CATEGORICAL_FEATURES = ["Brand", "Model", "Fuel Type", "Transmission Type", "Body Type"]
TARGET = "Price_EGP"


def load_and_clean_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
 
    df = df.dropna(subset=["Brand", "Model"]).reset_index(drop=True)

    df[TARGET] = (
        df[TARGET].astype(str)
        .str.replace("EGP", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)


    df["Kilometers"] = pd.to_numeric(df["Kilometers"], errors="coerce")
    df["Kilometers"] = df["Kilometers"].fillna(df["Kilometers"].median())

 
    if df["Fuel Type"].isnull().any():
        df["Fuel Type"] = df["Fuel Type"].fillna(df["Fuel Type"].mode()[0])

   
    if df["Body Type"].isnull().any():
        df["Body Type"] = df["Body Type"].fillna(df["Body Type"].mode()[0])

    if df["Engine Capacity (CC)"].isnull().any():
        knn_cols = ["Kilometers", "Year", "Engine Capacity (CC)", TARGET]
        imputer = KNNImputer(n_neighbors=5)
        imputed = imputer.fit_transform(df[knn_cols])
        df["Engine Capacity (CC)"] = imputed[:, knn_cols.index("Engine Capacity (CC)")]

    return df


def build_pipeline() -> Pipeline:
    """يبني الـ Pipeline: Encoding للأعمدة النصية + موديل RandomForest."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
             CATEGORICAL_FEATURES),
            ("num", "passthrough", NUMERIC_FEATURES),
        ]
    )
    model = RandomForestRegressor(n_estimators=15, max_depth=12,random_state=42,min_samples_leaf=5, max_features='sqrt')
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train_model(df: pd.DataFrame):

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    metrics = {
        "train_mae": mean_absolute_error(y_train, y_train_pred),
        "train_mse": mean_squared_error(y_train, y_train_pred),
        "train_r2": r2_score(y_train, y_train_pred),
        "test_mae": mean_absolute_error(y_test, y_test_pred),
        "test_mse": mean_squared_error(y_test, y_test_pred),
        "test_r2": r2_score(y_test, y_test_pred),
    }
    return pipeline, metrics


def print_overfitting_report(metrics: dict):
   
    r2_gap = metrics["train_r2"] - metrics["test_r2"]

    print("\n" + "=" * 55)
    print(" Train vs Test")
    print("=" * 55)
    print(f"{'Train':>30}{'Test':>18}")
    print(f"{'R2':<12}{metrics['train_r2']:>18.4f}{metrics['test_r2']:>18.4f}")
    print(f"{'MAE':<12}{metrics['train_mae']:>18,.0f}{metrics['test_mae']:>18,.0f}")
    print(f"{'MSE':<12}{metrics['train_mse']:>18,.0f}{metrics['test_mse']:>18,.0f}")
    print("-" * 55)
    print(f"Difference between R2 scores = {r2_gap:.4f}")

    if r2_gap > 0.15:
        print("There is Overfitting ")
    elif r2_gap > 0.07:
        print("There is simple overfitting")
    else:
        print("No overfitting ")
    print("=" * 55 + "\n")


class CarPricePredictorApp:
    """واجهة Tkinter للتنبؤ بسعر العربية."""

    def __init__(self, root: tk.Tk, pipeline: Pipeline, df: pd.DataFrame, metrics: dict):
        self.root = root
        self.pipeline = pipeline
        self.df = df

        root.title("توقع سعر العربية - Car Price Predictor")
        root.geometry("430x520")
        root.resizable(False, False)

        style = ttk.Style()
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10, "bold"))

        main = ttk.Frame(root, padding=20)
        main.pack(fill="both", expand=True)

        title = ttk.Label(
            main, text="أدخل بيانات العربية", font=("Segoe UI", 14, "bold")
        )
        title.grid(row=0, column=0, columnspan=2, pady=(0, 15))

       
        brands = sorted(df["Brand"].unique().tolist())
        fuels = sorted(df["Fuel Type"].unique().tolist())
        transmissions = sorted(df["Transmission Type"].unique().tolist())
        bodies = sorted(df["Body Type"].unique().tolist())
        self.brand_to_models = (
            df.groupby("Brand")["Model"].unique().apply(lambda x: sorted(x.tolist()))
        )

        row = 1

        ttk.Label(main, text="الماركة (Brand):").grid(row=row, column=0, sticky="w", pady=6)
        self.brand_var = tk.StringVar(value=brands[0])
        self.brand_combo = ttk.Combobox(
            main, textvariable=self.brand_var, values=brands, state="readonly", width=22
        )
        self.brand_combo.grid(row=row, column=1, pady=6)
        self.brand_combo.bind("<<ComboboxSelected>>", self.update_models)
        row += 1

        ttk.Label(main, text="الموديل (Model):").grid(row=row, column=0, sticky="w", pady=6)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(main, textvariable=self.model_var, state="readonly", width=22)
        self.model_combo.grid(row=row, column=1, pady=6)
        row += 1

        ttk.Label(main, text="سنة الصنع (Year):").grid(row=row, column=0, sticky="w", pady=6)
        self.year_var = tk.StringVar(value="2015")
        ttk.Entry(main, textvariable=self.year_var, width=24).grid(row=row, column=1, pady=6)
        row += 1

        ttk.Label(main, text="الكيلومترات (Kilometers):").grid(row=row, column=0, sticky="w", pady=6)
        self.km_var = tk.StringVar(value="80000")
        ttk.Entry(main, textvariable=self.km_var, width=24).grid(row=row, column=1, pady=6)
        row += 1

        ttk.Label(main, text="سعة المحرك CC:").grid(row=row, column=0, sticky="w", pady=6)
        self.cc_var = tk.StringVar(value="1600")
        ttk.Entry(main, textvariable=self.cc_var, width=24).grid(row=row, column=1, pady=6)
        row += 1

        ttk.Label(main, text="نوع الوقود (Fuel Type):").grid(row=row, column=0, sticky="w", pady=6)
        self.fuel_var = tk.StringVar(value=fuels[0])
        ttk.Combobox(
            main, textvariable=self.fuel_var, values=fuels, state="readonly", width=22
        ).grid(row=row, column=1, pady=6)
        row += 1

        ttk.Label(main, text="ناقل الحركة (Transmission):").grid(row=row, column=0, sticky="w", pady=6)
        self.trans_var = tk.StringVar(value=transmissions[0])
        ttk.Combobox(
            main, textvariable=self.trans_var, values=transmissions, state="readonly", width=22
        ).grid(row=row, column=1, pady=6)
        row += 1

        ttk.Label(main, text="شكل الهيكل (Body Type):").grid(row=row, column=0, sticky="w", pady=6)
        self.body_var = tk.StringVar(value=bodies[0])
        ttk.Combobox(
            main, textvariable=self.body_var, values=bodies, state="readonly", width=22
        ).grid(row=row, column=1, pady=6)
        row += 1

        predict_btn = ttk.Button(main, text="توقع السعر", command=self.predict)
        predict_btn.grid(row=row, column=0, columnspan=2, pady=(18, 8), ipadx=10, ipady=4)
        row += 1

        self.result_var = tk.StringVar(value="")
        ttk.Label(
            main, textvariable=self.result_var, font=("Segoe UI", 13, "bold"), foreground="#0a7d2c"
        ).grid(row=row, column=0, columnspan=2, pady=6)
        row += 1

        metrics_text = (
            f"Test: R² = {metrics['test_r2']:.2f} | MAE ≈ {metrics['test_mae']:,.0f} EGP   |   "
            f"Train: R² = {metrics['train_r2']:.2f} | MAE ≈ {metrics['train_mae']:,.0f} EGP"
        )
        ttk.Label(main, text=metrics_text, font=("Segoe UI", 8), foreground="#666").grid(
            row=row, column=0, columnspan=2, pady=(10, 0)
        )

        self.update_models()

    def update_models(self, event=None):
        brand = self.brand_var.get()
        models = self.brand_to_models.get(brand, [])
        self.model_combo["values"] = models
        if models:
            self.model_var.set(models[0])

    def predict(self):
        try:
            year = int(self.year_var.get())
            km = float(self.km_var.get())
            cc = float(self.cc_var.get())
        except ValueError:
            messagebox.showerror("خطأ في الإدخال", "من فضلك أدخل قيم رقمية صحيحة للسنة والكيلومترات وسعة المحرك.")
            return

        if not self.model_var.get():
            messagebox.showerror("خطأ في الإدخال", "من فضلك اختر موديل العربية.")
            return

        row = pd.DataFrame([{
            "Brand": self.brand_var.get(),
            "Model": self.model_var.get(),
            "Fuel Type": self.fuel_var.get(),
            "Transmission Type": self.trans_var.get(),
            "Body Type": self.body_var.get(),
            "Kilometers": km,
            "Year": year,
            "Engine Capacity (CC)": cc,
        }])

        price = self.pipeline.predict(row)[0]
        self.result_var.set(f"السعر المتوقع: {price:,.0f} EGP")


def main():
    print("Reading data and train mode ..")
    df = load_and_clean_data(CSV_PATH)
    pipeline, metrics = train_model(df)
    print_overfitting_report(metrics)

    root = tk.Tk()
    CarPricePredictorApp(root, pipeline, df, metrics)
    root.mainloop()


if __name__ == "__main__":
    main()