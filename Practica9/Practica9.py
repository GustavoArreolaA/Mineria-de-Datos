import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# 1. Carga de datos
df = pd.read_csv('SeoulBikeData_Limpio.csv')

# 2. Preparación del texto

texto = " ".join(review for review in df.astype(str))

# 3. Configuración de la nube de palabras

wordcloud = WordCloud(
    width=800, 
    height=400, 
    background_color='black',
    colormap='viridis', 
    min_font_size=10,
    random_state=42
).generate(texto)

# 4. Nube de palabras
fig = plt.figure(figsize=(8, 8))
fig.patch.set_facecolor('black')
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off") 
plt.title('Nube de Palabras: Dataset Seoul Bike Data', fontsize=15, color = 'white')
plt.show()