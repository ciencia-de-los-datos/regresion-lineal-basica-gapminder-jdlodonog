"""
Regresión Lineal Univariada
-----------------------------------------------------------------------------------------

En este laboratio se construirá un modelo de regresión lineal univariado.

"""
import numpy as np
import pandas as pd


def pregunta_01():
    """
    En este punto se realiza la lectura de conjuntos de datos.
    Complete el código presentado a continuación.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv(
    "gm_2008_region.csv",
    sep=",",  # separador de campos
    thousands=None,  # separador de miles para números
    decimal=".",  # separador de los decimales para números
    )

    print(df.head())
    # Asigne la columna "life" a `y` y la columna "fertility" a `X`
    y = df["life"].values
    X = df["fertility"].values

    #print(y)
    #print(X)

    # Imprima las dimensiones de `y`
    print(y.shape)
    #.ndim

    # Imprima las dimensiones de `X`
    print(X.shape)

    # Transforme `y` a un array de numpy usando reshape
    y_reshaped = y.reshape(-1, 1)

    # Trasforme `X` a un array de numpy usando reshape
    X_reshaped = X.reshape(-1, 1)

    # Imprima las nuevas dimensiones de `y`
    print(y_reshaped.shape)

    # Imprima las nuevas dimensiones de `X`
    print(X_reshaped.shape)


def pregunta_02():
    """
    En este punto se realiza la impresión de algunas estadísticas básicas
    Complete el código presentado a continuación.
    """
    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv(
    "gm_2008_region.csv",
    sep=",",  # separador de campos
    thousands=None,  # separador de miles para números
    decimal=".",  # separador de los decimales para números
    )
    # Imprima las dimensiones del DataFrame
    print(df.size)

    # Imprima la correlación entre las columnas `life` y `fertility` con 4 decimales.
    print(df.corr())

    # Imprima la media de la columna `life` con 4 decimales.
    print(df.mean())

    # Imprima el tipo de dato de la columna `fertility`.
    print(df["fertility"].dtypes)

    # Imprima la correlación entre las columnas `GDP` y `life` con 4 decimales.
    print((df['GDP'].corr(df['life'])))


def pregunta_03():
    """
    Entrenamiento del modelo sobre todo el conjunto de datos.
    Complete el código presentado a continuación.
    """


    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv(
    "gm_2008_region.csv",
    sep=",",  # separador de campos
    thousands=None,  # separador de miles para números
    decimal=".",  # separador de los decimales para números
    )

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility =df["fertility"].values

    # Asigne a la variable los valores de la columna `life`
    y_life = df["life"].values
    # Importe LinearRegression
    from sklearn.linear_model import LinearRegression

    # Cree una instancia del modelo de regresión lineal
    reg = LinearRegression()
    
    print("maximo_ minimo")

    print(max(X_fertility))
    print(min(X_fertility))

    # Cree El espacio de predicción. Esto es, use linspace para crear
    # un vector con valores entre el máximo y el mínimo de X_fertility
    prediction_space = np.linspace(
        max(X_fertility),
        min(X_fertility),
    ).reshape(-1, 1)

    #redmincionando las variables para que no muestre el error
    X_reshaped = X_fertility.reshape(-1, 1)
    y_reshaped = y_life.reshape(-1, 1)

    #NOTA: Receurde que se tiene que comvertir a array para poder usar el reshape
    # Entrene el modelo usando X_fertility y y_life
    reg.fit(X_reshaped, y_reshaped)
    # reg.fit(X_fertility, y_life

    # Compute las predicciones para el espacio de predicción
    y_pred = reg.predict(prediction_space)

    # Imprima el R^2 del modelo con 4 decimales
    print(reg.score(X_reshaped, y_life.reshape(-1,1).round(4)))



def pregunta_04():
    """
    Particionamiento del conjunto de datos usando train_test_split.
    Complete el código presentado a continuación.
    """

    # Importe LinearRegression
    # Importe train_test_split
    # Importe mean_squared_error
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error


    # Lea el archivo `gm_2008_region.csv` y asignelo al DataFrame `df`
    df = pd.read_csv(
    "gm_2008_region.csv",
    sep=",",  # separador de campos
    thousands=None,  # separador de miles para números
    decimal=".",  # separador de los decimales para números
    )

    # Asigne a la variable los valores de la columna `fertility`
    X_fertility = df["fertility"].values

    # Asigne a la variable los valores de la columna `life`
    y_life = df["life"].values

    # Divida los datos de entrenamiento y prueba. La semilla del generador de números
    # aleatorios es 53. El tamaño de la muestra de entrenamiento es del 80%
    (X_train, X_test, y_train, y_test,) =train_test_split(
        X_fertility,
        y_life,
        test_size=0.8,
        random_state=53,
    )

    # Cree una instancia del modelo de regresión lineal
    linearRegression = LinearRegression()

    # Entrene el clasificador usando X_train y y_train
    linearRegression.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))

    # Pronostique y_test usando X_test
    y_pred = linearRegression.predict(X_test.reshape(-1,1))

    # Compute and print R^2 and RMSE
    print("R^2: {:6.4f}".format(linearRegression.score(X_test.reshape(-1,1), y_test.reshape(-1,1))))
    rmse = np.sqrt(mean_squared_error(y_test, y_pred ))
    print("Root Mean Squared Error: {:6.4f}".format(rmse))
