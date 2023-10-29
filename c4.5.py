from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from cancer import X_normalized_df, y 

X_train, X_test, y_train, y_test = train_test_split(X_normalized_df, y, test_size=0.2, random_state=42)

c45_model = DecisionTreeClassifier(criterion='entropy')  

c45_model.fit(X_train, y_train)

y_pred = c45_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Acurácia do Modelo C4.5 (J48):", accuracy)
