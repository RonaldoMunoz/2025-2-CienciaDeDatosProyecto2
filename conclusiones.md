# Proyecto 2 y 3: Introducción a la ciencia de datos.


- Michael Rodriguez Arana
- Yeifer Ronaldo Muñoz Valencia
- Juan Carlos Rojas Quintero

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


### DESPUES DE VER LAS GRAFICAS PASAMOS A LA ETAPA DE LIMPIEZA Y NORMALIZACION DE DATOS

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