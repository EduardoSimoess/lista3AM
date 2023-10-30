from scipy import stats
import numpy as np
from c45 import acuracia_c45, precisao_c45, revocacao_c45
from id3 import acuracia_id3, precisao_id3, revocacao_id3
from svm import acuracia_svm, precisao_svm, revocacao_svm
from random_f import acuracia_random_forest, precisao_random_forest, revocacao_random_forest

c45 = [acuracia_c45, precisao_c45, revocacao_c45]
id3 = [acuracia_id3, precisao_id3, revocacao_id3]
svm = [acuracia_svm, precisao_svm, revocacao_svm]
random_forest = [acuracia_random_forest, precisao_random_forest, revocacao_random_forest]
print(c45, id3, svm, random_forest)


acuracias = [svm, random_forest, c45, id3]

for i in range(len(acuracias)):
    for j in range(i+1, len(acuracias)):
        modelo1 = acuracias[i]
        modelo2 = acuracias[j]
        _, p_valor = stats.ttest_ind(modelo1, modelo2)
        
        if not np.isnan(p_valor):
            if p_valor < 0.05:
                print(f"Diferença significativa entre {i} e {j} (p-valor: {p_valor})")
            else:
                print(f"Nenhuma diferença significativa entre {i} e {j} (p-valor: {p_valor})")
        else:
            print(f"Problema com o p-valor para {i} e {j} (p-valor: {p_valor})")
