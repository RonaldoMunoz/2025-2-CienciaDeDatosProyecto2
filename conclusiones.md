# Proyecto 2 y 3: Introducción a la ciencia de datos.


- Michael Rodriguez Arana
- Yeifer Ronaldo Muñoz Valencia
- Juan Carlos Rojas Quintero

#### HALLAZGOS OBTENIDOS

### ANALISIS DESCRIPTIVO DEL DATASET

En la primera parte del proyecto nos encontramos con el analicis descriptivo del dataset, donde se cargo el dataset en python usando pandas y se generaron visualizaciones exploratorias con Matplotlib y Seaborn

Primeramnete se realizó una adecuada carga y preparación del conjunto de datos avocado.csv, eliminando columnas innecesarias como Unnamed: 0 y convirtiendo el campo Date a un formato de fecha reconocible por Python. Este proceso permitió garantizar la coherencia y calidad de los datos antes del análisis exploratorio,

## Gráfico 1: Distribución del Precio Promedio

El histograma evidenció que la mayoría de los precios del aguacate se concentran entre 1.00 y 1.50 USD, con pocos casos que superan los 2.00 USD. La distribución presenta un sesgo positivo, lo cual indica que existen registros con precios más altos, aunque poco frecuentes. Esta característica podría influir en modelos predictivos o comparativos, por lo que se recomienda considerar normalización o transformaciones estadísticas en etapas posteriores.

## Gráfico 2: Comparación de Precios por Tipo de Aguacate

El diagrama de caja mostró que los aguacates orgánicos tienen un precio promedio significativamente mayor que los convencionales. Además, presentan una mayor variabilidad y más valores atípicos (outliers), lo que refleja la inestabilidad del mercado de productos orgánicos y la influencia de factores externos como la demanda o la disponibilidad.
Por otro lado, los aguacates convencionales exhiben precios más estables y concentrados, lo cual sugiere una oferta más constante y una menor sensibilidad a cambios de mercado.

## Gráfico 3: Mapa de Calor de Correlación de Variables Numéricas

El mapa de calor permitió identificar correlaciones significativas dentro del dataset.

Se observó una fuerte correlación positiva (≈ 0.97 a 0.99) entre las variables relacionadas con el volumen y las diferentes categorías de bolsas (Total Volume, 4046, 4225, Total Bags, etc.), lo que indica que los aumentos en el volumen total de ventas están estrechamente vinculados al incremento en la cantidad de unidades vendidas por tipo.

En cambio, el precio promedio (AveragePrice) presenta una correlación negativa moderada con el volumen total (-0.19), sugiriendo que, en general, a mayor volumen de ventas, menor precio promedio, lo cual concuerda con la lógica de oferta y demanda del mercado.

## Gráfico 4: Evolución del Precio Promedio (2015-2018)

La serie temporal de precios promedio entre 2015 y 2018 evidencia una variación estacional y cíclica. Se observan incrementos pronunciados en ciertos periodos, especialmente hacia mediados de 2017, donde los precios alcanzaron su punto máximo cercano a $1.85 USD. Posteriormente, se produce una tendencia descendente. Este comportamiento podría explicarse por fluctuaciones en la producción, la exportación o la estacionalidad del cultivo del aguacate.

## Gráfico 5: Distribución de Variables de Volumen (Datos Crudos)

Vemos la frecuencia de las ventas. Todos los gráficos están "aplastados" contra la izquierda, esto indica que la mayoría de las ventas son de volumen bajo/medio, pero existen unos pocos valores atípico de ventas masivas que distorsionan toda la escala, los modelos de regresión lineal funcionan muy mal con datos tan sesgados, por lo tanto debemos transformar estos datos.

## Gráfico 6: Distribución de Variables de Volumen (Log-Transformadas)

Los mismos histogramas del Gráfico anterior, pero después de aplicarles una transformación logarítmica, al aplicar el logaritmo, la escala se "comprime". Las distribuciones que antes estaban aplastadas ahora se asemejan mucho más a una distribución normal. Los outliers ya no dominan el gráfico, esto nos ayuda a entender que en la fase 2 no debemos eliminar estos valores atipicos que son ventas reales, sino que debemos aplicarles una transformación logaritmica.


### LIMPIEZA Y NORMALIZACION DE DATOS

Donde primeramente en esta parte del proceso se realizó la transformación de la variable temporal “Date” con el fin de hacerla útil para el análisis y los modelos de aprendizaje automático. 
Primero, la columna Date fue convertida al tipo de dato datetime, lo que permitió extraer información temporal relevante. A partir de ella, se generaron cuatro nuevas variables numéricas:
 - year_date: representa el año.
 - month: indica el mes (de 1 a 12).
 - day: muestra el día del mes.
 - week: corresponde al número de la semana del año según el calendario ISO.

Finalmente, se eliminó la columna original Date, ya que su formato de texto no es directamente interpretable por los modelos.
En conclusión, esta conversión permitió incorporar la dimensión temporal en forma numérica, facilitando la detección de patrones estacionales, tendencias o variaciones a lo largo del tiempo, algo esencial para mejorar el rendimiento y la interpretación de los modelos predictivos.

## Revisión y tratamiento de valores nulos dentro del conjunto de datos.

Primero, mediante la instrucción df.isna().sum(), se verificó la presencia de valores faltantes en cada columna, encontrándose que no existían datos nulos, sin embargo, para garantizar la consistencia y robustez del preprocesamiento, se aplicó una regla general de imputación preventiva en caso de que aparezcan valores faltantes en futuras actualizaciones del dataset:
 - En variables numéricas, los valores nulos serían reemplazados por la mediana de la columna.
 - En variables categóricas, se usaría la moda (el valor más frecuente).
En conclusión, esta revisión asegura que el conjunto de datos se mantenga libre de valores faltantes y, en caso de que surjan, se impute de forma coherente, evitando errores en los modelos y manteniendo la estabilidad de los resultados analíticos.

## detección y el tratamiento de valores atípicos (outliers) en las variables numéricas del conjunto de datos.

Para cada columna numérica, se calcularon los cuartiles (Q1 y Q3) y el rango intercuartílico (IQR = Q3 - Q1). A partir de estos, se establecieron los límites inferior y superior para definir los valores considerados atípicos:
 - Límite inferior: Q1 - 1.5 * IQR
 - Límite superior: Q3 + 1.5 * IQR
En lugar de eliminar las observaciones extremas, se aplicó una winsorización, es decir, se ajustaron los valores fuera de los límites hacia los valores límite más cercanos:
 - Valores menores al límite inferior se reemplazaron por el propio límite inferior.
 - Valores mayores al límite superior se reemplazaron por el límite superior.
En conclusión, este procedimiento permitió reducir la influencia de valores extremos sobre los modelos estadísticos o de machine learning, sin perder información al eliminar filas. De este modo, se preservó la integridad del conjunto de datos y se mejoró la estabilidad y fiabilidad del análisis posterior.

## codificación de variables categóricas mediante el uso de variables dummy (o indicadoras).

Se aplicó la función pd.get_dummies(df, drop_first=True) sobre las columnas no numéricas (por ejemplo, type o region), transformando cada categoría en una nueva columna binaria con valores 0 o 1, que indican la ausencia o presencia de dicha categoría. El parámetro drop_first=True se incluyó para evitar la multicolinealidad en modelos de regresión, ya que elimina una de las categorías de referencia por cada variable, manteniendo la información sin redundancias.
En conclusión, este paso permitió convertir todas las variables del conjunto de datos en formato numérico, requisito esencial para el entrenamiento de la mayoría de los modelos de machine learning, preservando la información categórica de forma adecuada y estadísticamente estable.

## estandarización de las variables predictoras

Primero, se separó la variable objetivo (AveragePrice) del conjunto de datos, ya que no debe ser transformada al representar el valor real a predecir. Luego, se identificaron las variables numéricas (excluyendo las dummies de regiones) que serían sometidas al proceso de escalado.
Para la estandarización se empleó la clase StandardScaler de scikit-learn, la cual transforma cada variable restando su media y dividiéndola por su desviación estándar. De esta forma, todas las variables quedan con una media cercana a 0 y una desviación estándar igual a 1.
El escalador se ajustó únicamente con los datos de entrenamiento (X_train), evitando la fuga de información (data leakage) hacia el conjunto de prueba. Posteriormente, se aplicó la misma transformación tanto a X_train como a X_test, manteniendo la coherencia en la escala de los datos.

### IMPLEMENTACION DE MODELOS PREDICTIVOS

## Regresion LIneal

El proceso siguió una metodología estructurada compuesta por las siguientes etapas:

**Carga de los datos:** se importaron los conjuntos de entrenamiento y prueba ya procesados, asegurando que todos los valores estuvieran listos para ser utilizados por el modelo.

**Separación de variables:** se distinguió la variable objetivo (AveragePrice) del conjunto de variables predictoras (X), necesarias para construir el modelo.

**Entrenamiento del modelo:** se entrenó un objeto de LinearRegression de la biblioteca scikit-learn, ajustando los coeficientes de la ecuación lineal que mejor se adapta a los datos de entrenamiento.

**Evaluación del desempeño:** se realizaron predicciones sobre el conjunto de prueba y se calcularon tres métricas clave:

**MAE (Mean Absolute Error):** mide el error promedio absoluto.

**RMSE (Root Mean Squared Error):** penaliza los errores grandes y refleja la precisión general del modelo.

**R² (Coeficiente de determinación):** indica qué proporción de la variabilidad del precio es explicada por las variables predictoras.

**Análisis de coeficientes:** se exportaron los coeficientes del modelo a un archivo CSV para identificar las variables con mayor influencia (positiva o negativa) sobre el precio del aguacate.

En conclusión, la Regresión Lineal permitió cuantificar la relación entre las características del aguacate y su precio promedio, proporcionando una primera aproximación interpretativa al problema. Este modelo sirve como línea base para comparar el desempeño con otros algoritmos más complejos en etapas posteriores.

## Ramdom Forest Regressor

En esta sección se implementó un modelo predictivo basado en Random Forest Regressor con el propósito de estimar el precio promedio del aguacate (AveragePrice) a partir de diversas variables explicativas, incluyendo tipo, región, volumen de ventas y componentes temporales.
El dataset empleado (avocado_final_clean.csv) ya había pasado por un proceso completo de preprocesamiento, que incluyó la eliminación de valores nulos, el tratamiento de outliers, la codificación de variables categóricas y la estandarización de las variables predictoras. Esto garantizó que los datos estuvieran en condiciones óptimas para el entrenamiento del modelo.
El Random Forest Regressor, perteneciente a la familia de los modelos de ensamble, combina múltiples árboles de decisión para mejorar la precisión y la estabilidad de las predicciones. Cada árbol es entrenado sobre una muestra aleatoria del conjunto de datos, y el resultado final se obtiene promediando las predicciones individuales, lo que reduce el riesgo de sobreajuste.

**Separación de variables y la división del conjunto de datos** se estableció la variable objetivo (y) como el precio promedio del aguacate (AveragePrice), mientras que las variables predictoras (X) correspondieron a todas las demás columnas del dataset. Esta separación es fundamental para el entrenamiento supervisado, ya que permite que el modelo aprenda la relación entre las características del aguacate y su precio, Posteriormente, se efectuó la división del dataset en dos subconjuntos con el objetivo de evaluar el rendimiento del modelo de forma objetiva:

 - Conjunto de entrenamiento (80%): utilizado para ajustar los parámetros del modelo.
 - Conjunto de prueba (20%): reservado para evaluar qué tan bien el modelo generaliza a datos no vistos.

La partición se realizó usando train_test_split con una semilla aleatoria (random_state=42) para garantizar la reproducibilidad de los resultados Y Finalmente, se configuró el modelo RandomForestRegressor con parámetros clave como:

 - n_estimators=200 → número de árboles en el bosque.
 - max_depth=None → sin límite de profundidad para permitir el crecimiento completo de los árboles.
 - random_state=42 → consistencia entre ejecuciones.
 - n_jobs=-1 → uso de todos los núcleos disponibles para acelerar el entrenamiento.

**Entrenamiento de modelo:** Los parámetros empleados fueron:
 - n_estimators=200 → número de árboles en el bosque.
 - max_depth=None → los árboles crecen sin límite de profundidad, lo que permite capturar relaciones complejas.
 - random_state=42 → asegura reproducibilidad de los resultados.
 - n_jobs=-1 → utiliza todos los núcleos del procesador para acelerar el entrenamiento.
El modelo fue ajustado con los datos de entrenamiento (X_train, y_train), completando exitosamente el proceso de aprendizaje.

**Evaluacion de desempeño:** Una vez entrenado, el modelo fue evaluado sobre el conjunto de prueba (X_test, y_test) mediante tres métricas fundamentales:
 - MAE (Mean Absolute Error): mide el error promedio absoluto entre los valores reales y las predicciones.
 - RMSE (Root Mean Squared Error): penaliza los errores grandes, reflejando la precisión general del modelo.
 - R² (Coeficiente de Determinación): indica qué tan bien el modelo explica la variabilidad del precio (1 = ajuste perfecto).
Los resultados obtenidos fueron:
MAE: 0.0852
RMSE: 0.1196
R²: 0.9084

El modelo revela que el tipo de aguacate es el factor clave en la determinación del precio, seguido por las características de empaque y volumen de ventas, mientras que los factores temporales tienen una influencia secundaria pero no despreciable.
Esto sugiere que tanto las estrategias de producción (tipo de producto) como las condiciones del mercado (volumen y época del año) impactan directamente en el valor final del aguacate.

## Red Neuronal

En esta etapa del análisis se implementó un modelo de Red Neuronal Multicapa (MLPRegressor) con el objetivo de predecir el precio promedio del aguacate (AveragePrice) a partir de diversas variables predictoras como volumen total, tipo de producto, bolsas por tamaño y datos temporales (semana, mes, año).

**carga y preparación de datos:**

Se utilizaron los archivos avocado_train_clean.csv y avocado_test_clean.csv, que ya habían sido previamente limpiados, estandarizados y codificados, Se separaron las variables predictoras (X) y la variable objetivo (y = AveragePrice), Se verificó que las variables estuvieran centradas en 0, confirmando que la estandarización fue correcta.

**Definición y entrenamiento del modelo:**

Se implementó un MLPRegressor con dos capas ocultas de tamaños (64, 32), función de activación ReLU, optimizador Adam, y un máximo de 1000 iteraciones, El modelo se entrenó con los datos de entrenamiento (X_train, y_train) hasta lograr convergencia.

**Evaluación del desempeño:**

Se realizaron predicciones sobre el conjunto de prueba (X_test) y se evaluaron las métricas:
MSE (Mean Squared Error): 0.0193
R² (Coeficiente de determinación): 0.8765
Los resultados muestran un bajo error promedio cuadrático y un alto nivel de explicación (≈ 87.6%) de la variabilidad del precio promedio del aguacate.

El modelo de Red Neuronal (MLPRegressor) logró resultados satisfactorios y consistentes, mostrando una capacidad predictiva alta (R² ≈ 0.88) y un error bajo (MSE ≈ 0.02).
Si bien su desempeño fue ligeramente inferior al Random Forest, representa un enfoque poderoso para capturar relaciones no lineales y complejas entre las variables del dataset.
Con un ajuste más fino de hiperparámetros, es probable que iguale o incluso supere el rendimiento de los métodos basados en árboles.


#### EFECTIVIDAD DE MODELOS PREDICTIVOS Y POSIBLES MEJORAS

## Regresion LIneal

# Efectividad
El modelo de Regresión Lineal presenta un desempeño aceptable pero no sobresaliente:

 - Explica el 60.8% de la variabilidad en el precio.

 - Muestra errores moderados (MAE ≈ 0.19).

 - Ofrece interpretabilidad alta, permitiendo identificar claramente la influencia de cada variable.

Sin embargo, comparado con modelos más complejos como Random Forest (R² ≈ 0.90) o Red Neuronal (R² ≈ 0.88), su capacidad predictiva es considerablemente menor, lo que sugiere que la relación entre las variables y el precio no es completamente lineal.

# Mejoras
 - Transformaciones de variables no lineales, Aplicar logaritmos, raíces o términos polinómicos (por ejemplo, TotalVolume², month²) para capturar relaciones curvas, Incluir interacciones entre variables (region × month, type × volume).

 - Regularización, Implementar variantes de la regresión lineal como Ridge o Lasso, que pueden mejorar la generalización reduciendo sobreajuste.

 - Selección de características (Feature Selection), Eliminar variables redundantes o poco relevantes y Utilizar técnicas automáticas como SelectKBest o Recursive Feature Elimination (RFE).

 - Modelos no lineales alternativos, Probar modelos más complejos como Random Forest, Gradient Boosting o Redes Neuronales, que ya demostraron mejor rendimiento.

## Random Forest

MAE (Mean Absolute Error) : 0.0852 , En promedio, el modelo se equivoca 0.085 unidades en la predicción del precio del aguacate.
RMSE (Root Mean Squared Error) : 0.1196 , Penaliza más los errores grandes; indica que las desviaciones típicas de las predicciones son de ~0.12.
R² (Coeficiente de Determinación) : 0.9084 ,  El modelo explica aproximadamente el 90.8% de la variabilidad del precio promedio del aguacate.

# Efectividad
 - El valor de R² = 0.9084 es excelente para un modelo de regresión, ya que implica que el modelo captura casi toda la relación entre las variables predictoras y la variable objetivo.

 - Los errores (MAE y RMSE) son bajos y consistentes, lo que indica buena estabilidad y precisión.

 - En términos prácticos: el modelo predice muy bien los precios del aguacate, aunque todavía hay un 9% de variabilidad no explicada (posiblemente por factores externos o ruido en los datos).

Random Forest es altamente efectivo y generaliza bien en los datos de prueba. Es una buena elección para este tipo de problema.

# Mejoras
 - GridSearchCV o RandomizedSearchCV para buscar la mejor combinación.

 - Validación cruzada (cv=5) para estimar mejor el rendimiento.
 
 - Eliminar o corregir outliers (valores atípicos de precio o volumen) que pueden distorsionar el entrenamiento.

 - Aumentar el tamaño de muestra, si tienes más datos disponibles.

 - Balancear las clases (si el tipo de aguacate está muy desbalanceado entre orgánico y convencional).

## Red Neuronal

MSE (Mean Squared Error) mide el error promedio cuadrático; cuanto menor sea, mejor.
→ 0.0193 indica que el error medio es bajo (buena predicción promedio).

R² (Coeficiente de determinación) mide cuánto de la variabilidad del precio es explicada por el modelo.
→ 0.8765 significa que la red explica el 87.6% de la variación del precio promedio, lo cual es muy bueno.

# Efectividad
Random Forest Regressor anterior (R² ≈ 0.9084), la red neuronal tiene un desempeño ligeramente inferior, pero sigue siendo sólido.

Random Forest	0.908	Mejor generalización, robusto a ruido
Red Neuronal (MLP)	0.877	Buen ajuste, pero podría mejorarse con tuning

# Mejoras
 - ajustar arquitectura de red 
  - Más neuronas o capas → (128, 64, 32) o (100, 50)
  - Activación 'tanh' si los datos están bien escalados.
  - Regularización: usar alpha=0.001 para evitar sobreajuste.

 - Normalización más precisa, Aunque ya escalaste tus variables, verifica que la media esté exactamente en 0 y la varianza en 1 (usa StandardScaler() correctamente antes del entrenamiento).

 ## 📊 Comparación de Modelos de Regresión

| Modelo | MAE ↓ | RMSE ↓ | MSE ↓ | R² ↑ | Interpretación |
|:--------|:-------:|:--------:|:-------:|:------:|:---------------|
| **Regresión Lineal** | 0.1903 | 0.2490 | 0.0620 | 0.6029 | Modelo base, solo capta relaciones lineales simples. |
| **Random Forest** | 0.0852 | 0.1196 | 0.0143 | 0.9084 | Excelente desempeño, capta relaciones no lineales y patrones complejos. |
| **Red Neuronal** | — | √0.0144 ≈ **0.1200** | 0.0144 | 0.9078 | Rendimiento muy similar a Random Forest, pero ligeramente menor. |

---

## 📈 Análisis Comparativo

1. **Precisión (MAE y RMSE):**
   - El **Random Forest** tiene los valores de error más bajos (**MAE = 0.0852**, **RMSE = 0.1196**), lo que indica que sus predicciones se acercan más a los valores reales.
   - La **Red Neuronal** muestra un rendimiento casi idéntico (**RMSE ≈ 0.1200**), lo que confirma su capacidad para capturar relaciones no lineales.
   - La **Regresión Lineal** presenta un error significativamente mayor, evidenciando que el problema **no es puramente lineal**.

2. **Capacidad explicativa (R²):**
   - Tanto **Random Forest (0.9084)** como la **Red Neuronal (0.9078)** explican más del **90% de la variabilidad** en los datos.
   - La **Regresión Lineal (0.6029)** solo explica el 60%, quedando como modelo de referencia o línea base.

3. **Comportamiento general:**
   - El **Random Forest** ofrece un **equilibrio excelente** entre precisión, estabilidad y facilidad de entrenamiento.
   - La **Red Neuronal** logra un rendimiento competitivo, aunque requiere mayor ajuste y tiempo de entrenamiento.
   - La **Regresión Lineal** es útil para interpretación y análisis de coeficientes, pero no para máxima precisión predictiva.

---

## 🧠 Conclusión General

> Tras la evaluación de los tres modelos de regresión sobre el dataset de precios de aguacates, se concluye que el **Random Forest** es el modelo con **mejor desempeño global**, alcanzando un **R² de 0.9084**, un **MAE de 0.0852** y un **RMSE de 0.1196**.  
> 
> Esto indica que el modelo explica más del **90% de la variabilidad** en los precios promedio con errores bajos y consistentes.  
> 
> La **Red Neuronal** obtuvo resultados casi equivalentes (R² = 0.9078), mientras que la **Regresión Lineal (R² = 0.6029)** mostró un rendimiento inferior al no capturar las relaciones no lineales del problema.  
> 
> En conclusión, el **Random Forest** se posiciona como el modelo más adecuado por su **alta precisión, robustez y buena capacidad de generalización**, sin requerir ajustes tan complejos como las redes neuronales.
