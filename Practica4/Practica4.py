import pandas as pd
from scipy import stats
import scikit_posthocs as sp

df = pd.read_csv('SeoulBikeData_Limpio.csv')

#Creación de la variable categórica "partes_dia"
def categorizar_hora(hora):
    
    if 6 <= hora <= 10: 
        return 'Mañana'
    
    elif 11 <= hora <= 16: 
        return 'Mediodía'
    
    elif 17 <= hora <= 21: 
        return 'Tarde'
    
    else: 
        return 'Noche/Madrugada'

df['partes_dia'] = df['hour'].apply(categorizar_hora)

#Preparación de los grupos 
grupos_prueba = [df[df['partes_dia'] == m]['rented_bike_count'] for m in df['partes_dia'].unique()]

#Kruskal-Wallis
stat_prueba, p_value = stats.kruskal(*grupos_prueba)

print(f"Prueba para partes del día:")
print(f"Estadístico de Prueba: {stat_prueba:.4f}\nValor-P: {p_value:.4e}")

#Post-hoc si el valor-P es demasiado bajo
if p_value < 0.05:
    print("\nResultados del Post-hoc (Dunn):")
    posthoc = sp.posthoc_dunn(df, val_col='rented_bike_count', group_col='partes_dia', p_adjust='holm')
    print(posthoc)