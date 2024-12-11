import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, fetch_california_housing
from sklearn.model_selection import train_test_split, StratifiedKFold, LeaveOneOut
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler

# Función para cargar datasets predefinidos

def load_datasets():
    # Cargar Breast Cancer Dataset
    breast_cancer = load_breast_cancer()
    X_bc = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
    y_bc = pd.Series(breast_cancer.target, name="target")
    dataset1 = pd.concat([X_bc, y_bc], axis=1)

    # Cargar California Housing Dataset y usar una muestra
    california_housing = fetch_california_housing()
    X_ch = pd.DataFrame(california_housing.data, columns=california_housing.feature_names)
    y_ch = pd.cut(california_housing.target, bins=2, labels=[0, 1]).astype(int)  # Convertir en clasificación binaria y asegurar tipo entero
    y_ch = pd.Series(y_ch, name="target")
    sampled_indices = np.random.choice(X_ch.index, size=10000, replace=False)  # Reducir tamaño a 10,000 muestras
    X_ch_sampled = X_ch.loc[sampled_indices]
    y_ch_sampled = y_ch.loc[sampled_indices]
    dataset2 = pd.concat([X_ch_sampled, y_ch_sampled], axis=1)

    # Cargar Covertype Dataset desde OpenML
    covertype = fetch_openml(name="covertype", version=3, as_frame=True)
    X_cv = covertype.data
    y_cv = covertype.target.astype(int)  # Convertir etiquetas a tipo entero
    sampled_indices = np.random.choice(X_cv.index, size=10000, replace=False)  # Reducir tamaño a 10,000 muestras
    X_cv_sampled = X_cv.loc[sampled_indices]
    y_cv_sampled = y_cv.loc[sampled_indices]

    # Asegurar que todas las características son numéricas y escalar los datos
    scaler = StandardScaler()
    X_cv_sampled = pd.DataFrame(scaler.fit_transform(X_cv_sampled), columns=X_cv_sampled.columns)
    dataset3 = pd.concat([X_cv_sampled, y_cv_sampled.rename("target")], axis=1)

    return [
        ("Breast Cancer", dataset1),
        ("California Housing", dataset2),
        ("Covertype", dataset3)
    ]

# Función para preprocesar datos
def preprocess_data(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # Eliminar filas con valores nulos en X o y
    valid_indices = ~X.isnull().any(axis=1) & ~y.isnull()
    X = X[valid_indices]
    y = y[valid_indices]

    # Verificar y filtrar clases con muy pocos ejemplos
    class_counts = y.value_counts()
    valid_classes = class_counts[class_counts > 1].index  # Filtrar clases con más de 1 ejemplo
    X = X[y.isin(valid_classes)]
    y = y[y.isin(valid_classes)]

    return X, y

# Función para validar modelo usando diferentes métodos de validación
def validate_model(X, y):
    results = {}
    model = GaussianNB()

    # Hold-Out (70/30 Estratificado)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results["Hold-Out"] = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Confusion Matrix": confusion_matrix(y_test, y_pred)
    }

    # 10-Fold Cross-Validation Estratificado
    skf = StratifiedKFold(n_splits=10)
    accuracies = []
    conf_matrix_sum = np.zeros((len(np.unique(y)), len(np.unique(y))))

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        conf_matrix_sum += confusion_matrix(y_test, y_pred)

    results["10-Fold Cross-Validation"] = {
        "Accuracy": np.mean(accuracies),
        "Confusion Matrix": conf_matrix_sum.astype(int)
    }

    # Leave-One-Out
    loo = LeaveOneOut()
    accuracies = []
    conf_matrix_sum = np.zeros((len(np.unique(y)), len(np.unique(y))))

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
        conf_matrix_sum += confusion_matrix(y_test, y_pred, labels=np.unique(y))

    results["Leave-One-Out"] = {
        "Accuracy": np.mean(accuracies),
        "Confusion Matrix": conf_matrix_sum.astype(int)
    }

    return results

# Cargar y procesar datasets
datasets = load_datasets()
all_results = {}

for name, data in datasets:
    print(f"Processing {name}...")
    X, y = preprocess_data(data)
    results = validate_model(X, y)
    all_results[name] = results

# Mostrar resultados
for dataset_name, result in all_results.items():
    print(f"\nResults for {dataset_name}:")
    for method, metrics in result.items():
        print(f"  {method}:")
        print(f"    Accuracy: {metrics['Accuracy']:.4f}")
        print(f"    Confusion Matrix:\n{metrics['Confusion Matrix']}\n")