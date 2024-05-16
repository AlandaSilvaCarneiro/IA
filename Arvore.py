import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import  tree

nome_arquivo = 'ConsumoCo2.csv'


dados = pd.read_csv(nome_arquivo)

dados_v1 = pd.DataFrame({'tipo_Comb':dados["FUELTYPE"],'Consumo_Cidade':dados["FUELCONSUMPTION_CITY"],'Consumo_maximo':dados["FUELCONSUMPTION_COMB"]})

x=dados_v1[["Consumo_Cidade","Consumo_maximo"]]

mapeamento = {'Z': 1, 'X': 2, 'E': 3, 'D': 4}
y=dados_v1["tipo_Comb"].replace(mapeamento)


x_trainer, x_teste,y_trainer,y_test = train_test_split(x,y, test_size=0.33, random_state=42)



arv = tree.DecisionTreeClassifier()
arv.fit(x_trainer,y_trainer)
print(arv.score(x_trainer,y_trainer))