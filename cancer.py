from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle  # Importe a função shuffle

# fetch dataset
wine = fetch_ucirepo(id=17)

# data (as pandas dataframes)
X = wine.data.features
y = wine.data.targets

# Embaralhe aleatoriamente os dados mantendo a correspondência entre X e y
X, y = shuffle(X, y, random_state=42)

# Inicialize o scaler Min-Max
scaler = MinMaxScaler()

# Ajuste o scaler aos dados e transforme as features
X_normalized = scaler.fit_transform(X)

# Crie um novo DataFrame com as features normalizadas
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)

# Exiba as features normalizadas
#print("Features Normalizadas:")
#print(X_normalized_df.head())

# Exiba as classes (rótulos)
#print("Classes (rótulos):")
#print(y)
