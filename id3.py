from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

from cancer import X_normalized_df, y  

X_train, X_test, y_train, y_test = train_test_split(X_normalized_df, y, test_size=0.2, random_state=42)

id3_model = DecisionTreeClassifier(criterion='entropy', splitter='best') 

id3_model.fit(X_train, y_train)

y_pred = id3_model.predict(X_test)

acuracia_id3 = accuracy_score(y_test, y_pred)

precisao_id3 = precision_score(y_test, y_pred, pos_label='M')

revocacao_id3 = recall_score(y_test, y_pred, pos_label='M')

print("Acurácia do Modelo ID3:", acuracia_id3)
