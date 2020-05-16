import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# function to load treated boston dataset
# st.cache tag indicate that func must be stay in cache to speed up process
@st.cache
def get_data():
    return pd.read_csv("model/boston_data.tsv", sep='\t')

# function to train model with the best predict model
def train_model():
    data = get_data()

    x = data.drop("MEDV",axis=1)
    y = data["MEDV"]

    # tuning random forest regressor to get a better prediction
    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7, max_features=3)
    rf_regressor.fit(x, y)

    return rf_regressor

data = get_data()

model = train_model()

st.title("Data App - Prevendo Valores de Imóveis")

# subtitle
st.markdown("Este é um Data App utilizado para exibir a solução de Machine Learning para o problema de predição de valores de imóveis de Boston.")

# verificando o dataset
st.subheader("Selecionando apenas um pequeno conjunto de atributos")

# default attributes to show
defaultcols = ["RM","PTRATIO","LSTAT","MEDV"]

# define attributes by multiselect
cols = st.multiselect("Atributos", data.columns.tolist(), default=defaultcols)

# show dataframe top 10
st.dataframe(data[cols].head(10))


st.subheader("Distribuição de imóveis por preço")

# define range of values
range_values = st.slider("Faixa de preço", float(data.MEDV.min()), 150., (10.0, 100.0))

# filter data
datas_to_show = data[data['MEDV'].between(left=range_values[0],right=range_values[1])]

# plot data distribution
f = px.histogram(datas_to_show, x="MEDV", nbins=100, title="Distribuição de Preços")
f.update_xaxes(title="MEDV")
f.update_yaxes(title="Total Imóveis")
st.plotly_chart(f)


st.sidebar.subheader("Defina os atributos do imóvel para predição")

# map user data for each attribute
crim = st.sidebar.number_input("Taxa de Criminalidade", value=data.CRIM.mean())
indus = st.sidebar.number_input("Proporção de Hectares de Negócio", value=data.CRIM.mean())
chas = st.sidebar.selectbox("Faz limite com o rio?",("Sim","Não"))

# convert string input data to binary
chas = 1 if chas == "Sim" else 0

nox = st.sidebar.number_input("Concentração de óxido nítrico", value=data.NOX.mean())

rm = st.sidebar.number_input("Número de Quartos", value=1)

ptratio = st.sidebar.number_input("Índice de alunos para professores",value=data.PTRATIO.mean())

b = st.sidebar.number_input("Proporção de pessoas com descendencia afro-americana",value=data.B.mean())

lstat = st.sidebar.number_input("Porcentagem de status baixo",value=data.LSTAT.mean())

# insert button in screen
btn_predict = st.sidebar.button("Realizar Predição")

# check if button was press
if btn_predict:
    result = model.predict([[crim,indus,chas,nox,rm,ptratio,b,lstat]])
    st.subheader("O valor previsto para o imóvel é:")
    result = "US $ "+str(round(result[0]*10,2))
    st.write(result)