# ==============================================================================
# GENERACIÓN DEL DATASET SINTÉTICO - E-BUYSUR S.A.
# PROYECTO FINAL: MINERÍA DE DATOS APLICADA
# AUTORES: MORALES, GUSTAVO & PRADO, LEILA
# ==============================================================================

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

# 1. Definición de la semilla para garantizar la reproducibilidad exacta
np.random.seed(42)

# 2. Generación de la estructura de clústeres base (3 grupos naturales)
# Simula 1.000 clientes activos con 2 variables de comportamiento comercial
X_raw, y_true = make_blobs(
    n_samples=1000, 
    n_features=2, 
    centers=3, 
    cluster_std=0.50, 
    random_state=42
)

# 3. Interpolación y escalado a los rangos reales de negocio de E-BuySur
# Variable 1: Frecuencia de Compra Anual (Rango: 1 a 24 compras al año)
frecuencia = np.interp(X_raw[:, 0], (X_raw[:, 0].min(), X_raw[:, 0].max()), (1, 24)).astype(int)

# Variable 2: Monto Anual Gastado (Rango: $10.000 a $500.000 pesos argentinos)
monto = np.interp(X_raw[:, 1], (X_raw[:, 1].min(), X_raw[:, 1].max()), (10000, 500000))

# 4. Construcción del DataFrame consolidado
df_clientes = pd.DataFrame({
    'Id_Cliente': [f'EBS-{i:04d}' for i in range(1, 1001)],
    'Frecuencia_Compra': frecuencia,
    'Monto_Anual_Gastado': monto
})

# 5. Exportación a archivo de texto plano (CSV) exigido por la pauta
df_clientes.to_csv('dataset_clientes_ebuysur.csv', index=False)

print("¡Éxito! El archivo 'dataset_clientes_ebuysur.csv' ha sido generado correctamente.")
print(f"Total de registros exportados: {len(df_clientes)} clientes.\n")

# Muestra preliminar en consola de los primeros 5 registros
print("Vista previa del dataset:")
print(df_clientes.head())