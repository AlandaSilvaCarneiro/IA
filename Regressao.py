import pandas as pd
from sklearn.linear_model import  LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
nome_arquivo = 'ConsumoCo2.csv'


dados = pd.read_csv(nome_arquivo)



y=dados["FUELCONSUMPTION_COMB"]
x=dados["CO2EMISSIONS"]

novo_dados = pd.DataFrame({'Consumo_Comb': y, 'Co2Emissao': x})
x_trainer, x_teste,y_trainer,y_test = train_test_split(x,y, test_size=0.33, random_state=42)
reg = LinearRegression().fit(x_trainer.values.reshape(-1,1),y_trainer)
y_pred = reg.predict(x_teste.values.reshape(-1,1))


print(mean_squared_error(y_test,y_pred))
print((mean_absolute_error(y_test,y_pred)))


plt.scatter(y_test, y_pred, color='blue', label='Valores Previstos')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Linha de Regressão')  # Linha de regressão
plt.scatter(y_test, y_test, color='green', label='Valores Reais')  # Valores reais
plt.xlabel('Valores Reais')
plt.ylabel('Valores Previstos')
plt.title('Regressão Linear: Valores Reais vs. Valores Previstos')
plt.legend()
plt.show()







