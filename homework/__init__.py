import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import sys
import os

def create_and_save_estimator():
    data_path = "auto_mpg.csv" 
    try:
        dataset = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo en {data_path}")
        print(f"Directorio actual: {os.getcwd()}")
        sys.exit(1)

    dataset = dataset.dropna()
    dataset["Origin"] = dataset["Origin"].map(
        {1: "USA", 2: "Europe", 3: "Japan"},
    )

    y = dataset.pop("MPG").astype(int) 
    X = dataset.copy()

    print("Guardando CSV limpio en la raíz del proyecto...")
    
    cleaned_dataset_to_save = X.copy()
    cleaned_dataset_to_save["MPG"] = y 
    
    output_csv_path = "auto_mpg.csv"
    
    try:
        cleaned_dataset_to_save.to_csv(output_csv_path, index=False)
        print(f"CSV limpio guardado exitosamente en: {output_csv_path}")
    except Exception as e:
        print(f"Error al guardar el CSV limpio: {e}")

    numeric_features = [
        "Cylinders", 
        "Displacement", 
        "Horsepower", 
        "Weight", 
        "Acceleration", 
        "Model Year"
    ]
    categorical_features = ["Origin"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop"
    )

    model = RandomForestClassifier(random_state=42)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            ("classifier", model)
        ]
    )

    print("Entrenando el modelo (como clasificación)...")
    pipeline.fit(X, y)
    print("Entrenamiento completado.")

    y_pred_train = pipeline.predict(X)
    train_accuracy = accuracy_score(y, y_pred_train)
    
    output_file = "estimator.pickle"

    with open(output_file, "wb") as file:
        pickle.dump(pipeline, file)

if __name__ == "__main__":
    create_and_save_estimator()