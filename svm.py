from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score

from cancer import X_normalized_df, y  

X_train, X_test, y_train, y_test = train_test_split(X_normalized_df, y, test_size=0.2, random_state=42)

svm_model = SVC(kernel='poly')

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

acuracia_svm = accuracy_score(y_test, y_pred)

precisao_svm = precision_score(y_test, y_pred, pos_label='M')

revocacao_svm = recall_score(y_test, y_pred, pos_label='M')

print("Acurácia do SVM:", acuracia_svm)
