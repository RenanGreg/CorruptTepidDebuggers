import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import OneHotEncoder , StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error 
from tabulate import tabulate

data = {
  "Nome": ["Igor", "Clara", "Evelyn", "Olivia", "Tiago", "Beatriz", "Igor", "Zoe", "Igor", "Mariana", "Bruno", "Eduardo", "Vitor", "Helena", "Juliana", "Fernanda", "Evelyn",
           "Paulo", "Rafael", "Clara", "Evelyn", "Eduardo", "Xavier", "Fernanda", "Ursula", "Rafael", "Fernanda", "Fernanda", "Gabriel", "Igor"],
  
  
  "Idade": [22, 54, 32, 25, 55, 22, 37, 35, 35, 51, 32, 50, 42, 42, 28, 24, 46, 26, 44, 31, 37, 35, 38, 25, 49, 37, 47, 59, 29, 34],

  
  "Cidade": ["Florianópolis", "BeloHorizonte", "RiodeJaneiro", "BeloHorizonte", "Florianópolis", "PortoAlegre", "Curitiba", "RiodeJaneiro", "Curitiba", "SãoPaulo",
             "Florianópolis", "SãoPaulo", "PortoAlegre", "PortoAlegre", "SãoPaulo", "SãoPaulo", "BeloHorizonte", "PortoAlegre", "BeloHorizonte", "SãoPaulo", "SãoPaulo", 
             "Florianópolis", "SãoPaulo", "Florianópolis", "Florianópolis", "Curitiba", "SãoPaulo", "BeloHorizonte", "Florianópolis", "PortoAlegre"],
  
  
  "Profissao": ["Programador", "Gestor", "Engenheiro", "Enfermeiro", "Advogado", "Advogado", "Arquiteto", "Gestor", "Advogado", "Médico", "Médico", "Analista", "Professor",
                "Professor", "Advogado", "Gestor", "Professor", "Professor", "Engenheiro", "Gestor", "Designer", "Programador", "Analista", "Arquiteto", "Gestor", "Analista",
                "Designer", "Gestor", "Professor", "Analista"],
  
  
  "Salario": [7710, 4964, 5617, 4959, 4916, 8264, 5692, 8406, 8212, 4571, 7034, 10803, 6768, 7029, 4175, 7536, 10255, 9053, 4627, 10848, 8244, 11310, 8917, 10636, 9013,
              10143, 4474, 10374, 5411, 9998],

  
  "Anos de Experiencia": [1, 34, 12, 15, 22, 22, 16, 32, 20, 18, 21, 14, 30, 34, 5, 22, 21, 23, 31, 11, 20, 5, 10, 1, 2, 5, 9, 1, 23, 5]
}


df = pd.DataFrame(data) 

X = df.drop(columns=["Nome", "Salario"])
y = df["Salario"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) 


preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Idade", "Anos de Experiencia"]),
        ("cat", OneHotEncoder(), ["Cidade", "Profissao"])
    ]
)


pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(random_state=42))
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}") 


novos_dados = pd.DataFrame({
  "Idade": [30],
  "Cidade": ["SãoPaulo"],
  "Profissao": ["Programador"],
  "Anos de Experiencia": [5]
})

previsao = pipeline.predict(novos_dados)
print(f"Previsão de Salário: {previsao[0]:.2f}") 
print(tabulate(df, headers='keys', tablefmt='grid')) 



