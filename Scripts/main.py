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


df_menores = df[df["age"] < 18]

# Crear el gráfico de regresión lineal dentro de cada género
sns.lmplot(x='media_combinada', y='mental_health_score', data=df_menores, aspect=1.5, scatter_kws={'alpha':0.5}, hue="gender")

# Etiquetas y título
plt.xlabel('Horas de pantalla semanales')
plt.ylabel('Puntuación de salud mental')
plt.title('Relación entre tiempo de pantalla y salud mental en menores')

# Mostrar gráfico
plt.savefig("../Imagenes/Mental_health_score_gender.jpg")

sns.lmplot(data = df_menores,x= "social_media_hours", y= "mental_health_score");
plt.savefig("../Imagenes/Mental_health_score_teens.jpg")

sns.lmplot(data = df_menores,x= "social_media_hours", y= "mental_health_score", aspect=1.5, scatter_kws={'alpha':0.5}, hue="gender")

# Etiquetas y título
plt.xlabel('Horas de pantalla semanales')
plt.ylabel('Puntuación de salud mental')
plt.title('Relación entre tiempo de pantalla y salud mental en menores')

# Mostrar gráfico
plt.savefig("../Imagenes/Mental_health_score_teens_gender.jpg")



#-------------------------------------------------------------


df_3 = pd.read_csv("../Datasets/Mental_Health_lifestyle_Dataset.csv")
df_3.fillna("No", inplace=True)

df3 = df_3[["Sleep Hours", "Work Hours per Week", "Screen Time per Day (Hours)", "Happiness Score", "Social Interaction Score"]]

# Suponiendo que tu DataFrame se llama df
variables = ["Sleep Hours", "Work Hours per Week", "Screen Time per Day (Hours)", "Social Interaction Score"]
objetivo = "Happiness Score"

correlaciones = df3[variables].corrwith(df3[objetivo])

agrupado_condition = df_3.groupby('Mental Health Condition')["Happiness Score"].mean()

# Crear DataFrame correctamente
data = {
    "Condición": ["Ansiedad", "Bipolar", "Depresión", "Ninguna", "Estres post trauma"],
    "Happiness Score": [5.258121, 5.471553, 5.341552, 5.452437, 5.457692]
}
df_datacondition = pd.DataFrame(data)

# Graficar línea asegurando que el DataFrame se usa correctamente
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_datacondition, x="Condición", y="Happiness Score", marker="o", color="blue")
plt.title("Nivel de Felicidad según Condición")
plt.ylabel("Nivel de felicidad")
plt.xlabel("Condición")
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("../Imagenes/Mental_health_score_condition.jpg")
agrupado_stress = df_3.groupby('Stress Level')["Happiness Score"].mean()
data2 = {
    "Nivel de Stress": ["Alto", "Bajo", "Moderado"],
    "Happiness Score": [5.440419, 5.408631, 5.335354]
}
df_datacondition2 = pd.DataFrame(data2)

# Graficar línea asegurando que el DataFrame se usa correctamente
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_datacondition2, x="Nivel de Stress", y="Happiness Score", marker="o", color="blue")
plt.title("Nivel de Felicidad según estres")
plt.ylabel("Nivel de felicidad")
plt.xlabel("Nivel de Estrés")
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("../Imagenes/Mental_health_score_estres.jpg")

agrupado_dieta = df_3.groupby('Diet Type')["Happiness Score"].mean()
data3 = {
    "Tipo de comida": ["Equilibrada", "Comida Basura", "Keto","Vegana","Vegetariano"],
    "Happiness Score": [5.247680, 5.436264, 5.339616, 5.287086,5.664527]
}
df_datacondition3 = pd.DataFrame(data3)

# Graficar línea asegurando que el DataFrame se usa correctamente
plt.figure(figsize=(8, 6))
sns.lineplot(data=df_datacondition3, x="Tipo de comida", y="Happiness Score", marker="o", color="blue")
plt.title("Nivel de Felicidad según la alimentación")
plt.ylabel("Nota de la felicidad ")
plt.xlabel("Tipo de dieta")
plt.xticks(rotation=45)
plt.grid(True)
plt.savefig("../Imagenes/Mental_health_score_diet.jpg")



#-------------------------------------------------

df2 =pd.read_csv("../Datasets/survey.csv") 
df2_1 = df2[["family_history" , "mental_health_consequence", "work_interfere"]]
mapeo = {
    "No": 0,
    "Yes": 1,
    "Maybe": 0.5,
    "Often": 3,
    "Sometimes": 2,
    "Rarely": 1,
    "Never": 0
}
df2_1 = df2_1.replace(mapeo)
df2_1 = df2_1.replace(mapeo)
df_grouped = df2_1[df2_1["family_history"] == 1]
df_grouped2 = df2_1[df2_1["family_history"] == 0]

#------------------------------------

# Dentro de la agrupación anterior de los casos que si han tenido historial genético agrupo a su vez por las puntuaciones de la salud mental actual
mental_health_count = df_grouped[df_grouped["mental_health_consequence"] == 1].shape[0]
mental_health_count2 = df_grouped[df_grouped["mental_health_consequence"] == 0.5].shape[0]
mental_health_count3 = df_grouped[df_grouped["mental_health_consequence"] == 0].shape[0]

#------------------------------------

# Dentro de la agrupación anterior de los casos que no han tenido historial genético agrupo a su vez por las puntuaciones de la salud mental actual2
mental_health_count1 = df_grouped2[df_grouped2["mental_health_consequence"] == 1].shape[0]
mental_health_count21 = df_grouped2[df_grouped2["mental_health_consequence"] == 0.5].shape[0]
mental_health_count31 = df_grouped2[df_grouped2["mental_health_consequence"] == 0].shape[0]

total = mental_health_count3 + mental_health_count2 + mental_health_count
total2 = mental_health_count1 + mental_health_count21 + mental_health_count31
# Mostrar resultados


porcentaje_si = (mental_health_count / total) * 100
porcentaje_quizas = (mental_health_count2 / total) * 100
porcentaje_no = (mental_health_count3 / total) * 100 

porcentaje_si_1 = (mental_health_count1 / total2) * 100
porcentaje_quizas_1 = (mental_health_count21 / total2) * 100
porcentaje_no_1 = (mental_health_count31 / total2) * 100 








#calculo el total de las agrupaciones en funcion del historial

total = mental_health_count3 + mental_health_count2 + mental_health_count
total2 = mental_health_count1 + mental_health_count21 + mental_health_count31
# Mostrar resultados



df_grouped = df2_1.groupby(["family_history", "mental_health_consequence"]).size().reset_index(name="count")
# Mapear los valores numéricos a etiquetas descriptivas
df_grouped["family_history"] = df_grouped["family_history"].map({0: "No", 1: "Sí"})
df_grouped["mental_health_consequence"] = df_grouped["mental_health_consequence"].map({0.0: "No", 0.5: "Quizás", 1.0: "Sí"})
# Crear el gráfico de barras agrupadas
plt.figure(figsize=(8, 6))  # Ajustar tamaño si es necesario
sns.barplot(
    x="mental_health_consequence",
    y="count",
    hue="family_history",
    data=df_grouped,
    palette="Blues_d",
    order=["No", "Quizás", "Sí"]  # el orden de las categorías
)
# Configurar etiquetas y título
plt.xlabel("Respuestas", fontsize=12)
plt.ylabel("Cantidad", fontsize=12)
plt.title("Comparación de consecuencias negativas en salud mental", fontsize=14, pad=20)
# Personalizar leyenda
plt.legend(title="Historial Familiar", title_fontsize=12, fontsize=11)
# Ajustar diseño y mostrar gráfico
plt.tight_layout()

plt.savefig("../Imagenes/Mental_health_score_genetics.jpg")
