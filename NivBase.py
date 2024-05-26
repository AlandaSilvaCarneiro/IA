import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

nome_arquivo = 'ConsumoCo2.csv'

dados = pd.read_csv(nome_arquivo)



y=dados["FUELCONSUMPTION_COMB"]
x=dados["CO2EMISSIONS"]

novo_dados = pd.DataFrame({'Consumo_Comb': y, 'Co2Emissao': x})
X_train, X_test, y_train, y_test = train_test_split(x,
y, test_size=0.5, random_state=0)
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
print((X_test.shape[0], (y_test !=y_pred).sum()))