# Estados del MDP con K-Means

Este proyecto explora un conjunto de datos médicos mediante limpieza básica y
agrupamiento no supervisado. Se usan los valores de troponina y CK-MB para
formar cuatro grupos (estados) con K-Means. El objetivo es describir patrones
en los datos; los grupos no son diagnósticos ni sustituyen la evaluación de un
profesional de la salud.

## Qué se realizó

- Se revisaron valores nulos y filas duplicadas.
- Se excluyeron las filas cuya frecuencia cardiaca está fuera del rango de 30
  a 200 latidos por minuto.
- Se transformaron Troponin y CK-MB con logaritmo base 10 y se estandarizaron
  antes de aplicar K-Means con `K=4`.
- Para dar un orden descriptivo a los grupos, se ordenaron según la proporción
  de resultados `positive` y se les asignaron los nombres Bajo, Dudoso,
  Alto-Troponina y Alto-CKMB. Por eso, los nombres reflejan el perfil observado
  en este conjunto de datos, no categorías clínicas.
- Se calcularon perfiles por grupo y la puntuación silhouette para describir
  la separación obtenida.

## Requisitos

- Python 3
- `numpy`, `pandas`, `scikit-learn`, `joblib` y `matplotlib`

Desde la carpeta del proyecto, se puede crear un entorno e instalar las
dependencias así:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn joblib matplotlib
```

En Windows, se puede activar el entorno con
`.venv\Scripts\activate` en el símbolo del sistema o
`.venv\Scripts\Activate.ps1` en PowerShell.

## Cómo ejecutar

Mantén `Medicaldataset.csv` en la carpeta del proyecto y ejecuta:

```bash
python estados.py
```

El programa imprime un resumen de la limpieza, la puntuación silhouette y el
perfil de cada estado. También genera o reemplaza estos archivos:

- `dataset_con_estados.csv`: datos conservados con las columnas derivadas y el
  estado asignado.
- `perfil_estados.csv`: cantidad de registros, medianas de variables y
  porcentaje de resultados positivos por estado.
- `modelo_estados.joblib`: scaler, modelo K-Means y datos usados para mapear
  los grupos a sus nombres.

Para volver a generar las gráficas, después de ejecutar `estados.py`, ejecuta:

```bash
python grafica.py
```

## Gráficas

### Distribución de los estados

![Dispersión de troponina y CK-MB por estado](grafica_estados_dispersion.png)

Cada punto representa un registro y el color indica el estado asignado. Los
ejes muestran los valores transformados con logaritmo base 10; las marcas X
grandes indican el promedio de cada grupo en esa escala.

### Porcentaje de resultados positivos

![Proporción de resultados positivos por estado](grafica_estados_positivos.png)

La altura de cada barra muestra qué porcentaje de registros del estado tiene
`Result = positive`. Sobre cada barra se indica el tamaño del grupo. En estos
datos, los estados Alto-Troponina y Alto-CKMB tienen una proporción de positivos
mayor que los estados Bajo y Dudoso.

## Archivos principales

- `estados.py`: limpieza, transformación, agrupamiento y generación de salidas.
- `grafica.py`: generación de las dos gráficas a partir de
  `dataset_con_estados.csv`.
- `Medicaldataset.csv`: conjunto de datos de entrada.
