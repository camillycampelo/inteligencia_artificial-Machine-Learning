import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer #Dataset embutido no scikit-learn com informações sobre tumores malignos e benignos.

dados = load_breast_cancer()
x_dados = dados.data # Atributos/features
y_resultado = dados.target   # Classes (0: Maligno, 1: Benigno)
feature_names = dados.feature_names # Nomes das features

#dados.data: Matriz de atributos com 30 características numéricas.
#dados.target: Classes binárias indicando se o tumor é maligno (0) ou benigno (1).
#dados.feature_names: Nomes das características.

df_cancer = pd.DataFrame(x_dados, columns=feature_names)
df_cancer['target'] = y_resultado

df_cancer

import matplotlib.pyplot as plt

resultado = df_cancer['target'].value_counts()
rotulos = ['Maligno', 'Benigno']
plt.bar(rotulos, resultado)
plt.title('Distribuição dos dados')
plt.xlabel('Classe')
plt.ylabel('Quantidade')
plt.show()

#Como os dados estão relacionados?
matriz_correlacao = df_cancer.corr() #Gera a matriz de correlação, que mostra o grau de relação linear entre pares de atributos.
matriz_correlacao

plt.figure(figsize = (12,8))
plt.imshow(matriz_correlacao, cmap='coolwarm', interpolation='nearest') #coolwarm: Coloração para indicar intensidade da correlação.
plt.xticks(range(len(matriz_correlacao.columns)), matriz_correlacao.columns, rotation=90)
plt.yticks(range(len(matriz_correlacao.columns)), matriz_correlacao.columns)
plt.colorbar()

