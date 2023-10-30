from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from cancer import X_normalized_df, y 

X_train, X_test, y_train, y_test = train_test_split(X_normalized_df, y, test_size=0.2, random_state=42)

c45_model = DecisionTreeClassifier(criterion='entropy')  

c45_model.fit(X_train, y_train)

y_pred = c45_model.predict(X_test)

acuracia_c45 = accuracy_score(y_test, y_pred)

precisao_c45 = precision_score(y_test, y_pred, pos_label='M')

revocacao_c45 = recall_score(y_test, y_pred, pos_label='M')

print("Acurácia do Modelo C4.5 (J48):", acuracia_c45)
