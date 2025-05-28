import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../Datasets/digital_diet_mental_health.csv")
media_screen = df.groupby("gender")["daily_screen_time_hours"].mean()
df["total_usage_hours"] = df[
    [
        "phone_usage_hours",
        "laptop_usage_hours",
        "tablet_usage_hours",
        "tv_usage_hours",
        "social_media_hours",
        "work_related_hours",
    ]
].sum(axis=1)

bins = [0, 18, 30, 50, 100]  # Límites de los rangos
labels = ['Menor', 'Joven', 'Adulto', 'Mayor']  # Etiquetas para los rangos
df['Rango_Edad'] = pd.cut(df['age'], bins=bins, labels=labels)


df["media_combinada"]= (df["daily_screen_time_hours"] + df["total_usage_hours"])
sns.lmplot(x='media_combinada', y='mental_health_score', data=df, aspect=1.5, scatter_kws={'alpha':0.5},hue ="Rango_Edad")
plt.xlabel('Horas de pantalla Semanales')
plt.ylabel('Puntuación de salud mental')
plt.title('Relación entre tiempo de pantalla y salud mental')
plt.savefig("../Imagenes/Mental_health_score.jpg")