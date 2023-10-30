from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from cancer import X_normalized_df, y 

X_train, X_test, y_train, y_test = train_test_split(X_normalized_df, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42) 

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

acuracia_random_forest = accuracy_score(y_test, y_pred)

precisao_random_forest = precision_score(y_test, y_pred, pos_label='M')

revocacao_random_forest = recall_score(y_test, y_pred, pos_label='M')

print("Acurácia do Modelo Random Forest:", acuracia_random_forest)
