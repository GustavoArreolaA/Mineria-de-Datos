import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)
df = pd.read_csv('SeoulBikeData_Limpio.csv', encoding='latin-1')
df.info()

graficos = [
    {'tipo': 'hist', 'col': 'rented_bike_count', 'titulo': 'Histograma de Rentas'},
    {'tipo': 'pie', 'col': ['seasons_Autumn', 'seasons_Spring', 'seasons_Summer', 'seasons_Winter'], 'titulo': 'Distribución de rentas por Estaciones'},
    {'tipo': 'scatter', 'col': ['temperature', 'rented_bike_count'], 'titulo': 'Temperatura vs Rentas'},
    {'tipo': 'line', 'col': 'hour', 'titulo': 'Tendencia por Hora'},
    {'tipo': 'bar_mean', 'col': ['holiday', 'rented_bike_count'], 'titulo': 'Rentas: Festivo vs Laboral'}
]

estaciones = ["Spring", "Summer", "Autumn", "Winter"]


for g in graficos:
    plt.figure(figsize=(8, 4))
    
    if g['tipo'] == 'hist':
        sns.histplot(df[g['col']], kde=True, color='pink')
    elif g['tipo'] == 'pie':
        df[g['col']].value_counts().plot.pie(labels = estaciones, autopct='%1.1f%%', colors=['#27AE60','#F1C40F', '#8B4513', '#87CEEB'])
    elif g['tipo'] == 'scatter':
        sns.scatterplot(x=df[g['col'][0]], y=df[g['col'][1]], alpha=0.5, color='purple')
    elif g['tipo'] == 'line':
        df.groupby(g['col'])['rented_bike_count'].mean().plot()
        plt.ylabel('Mean of rented bikes')
    elif g['tipo'] == 'bar_mean':
        sns.barplot(x=df[g['col'][0]], y=df[g['col'][1]], estimator='sum', errorbar=None, palette=["#F1260F", '#27AE60'])
        plt.xticks([0, 1], ['No', 'Si'])

    plt.title(g['titulo'])
    plt.show()

