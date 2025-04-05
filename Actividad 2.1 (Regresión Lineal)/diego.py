################################################ Función para cargar un archivo como un dataframe #############################################################

def cargar_archivo(archivo):
    import pandas as pd
    import os
    extension = os.path.splitext(archivo)[1].lower()
    if extension == '.csv':
        df = pd.read_csv(archivo)
        return(df)
    elif extension == '.xlsx':
        df = pd.read_excel(archivo)
        return(df)
    elif extension == '.json':
        df = pd.read_json(archivo)
        return(df)
    elif extension == '.html':
        df = pd.read_html(archivo)
        return(df)
    else:
        raise ValueError(f'Formato de archivo no soportado: {extension}')
    
################################################ Funciones mean para sustitución de valores nulos #############################################################

def sust_prom(dataframe):
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    cuantitativas_con_nulos = dataframe.select_dtypes(include=['float64', 'int64','float','int'])
    cualitativas = dataframe.select_dtypes(include=['object', 'datetime','category'])
    cuantitativas = cuantitativas_con_nulos.fillna(round(cuantitativas_con_nulos.mean(), 1))
    Datos_sin_nulos = pd.concat([cuantitativas, cualitativas], axis=1)
    return(Datos_sin_nulos)

################################################ Funciones ffill para sustitución de valores nulos ############################################################

def sust_ffill(dataframe):
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    cuantitativas = dataframe.select_dtypes(include=['float64', 'int64','float','int'])
    cualitativas_con_nulos = dataframe.select_dtypes(include=['object', 'datetime','category'])
    cualitativas = cualitativas_con_nulos.fillna(method='ffill')
    Datos_sin_nulos = pd.concat([cuantitativas, cualitativas], axis=1)
    return(Datos_sin_nulos)

################################################ Funciones bfill para sustitución de valores nulos ############################################################

def sust_bfill(dataframe):
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    cuantitativas = dataframe.select_dtypes(include=['float64', 'int64','float','int'])
    cualitativas_con_nulos = dataframe.select_dtypes(include=['object', 'datetime','category'])
    cualitativas = cualitativas_con_nulos.fillna(method='bfill')
    Datos_sin_nulos = pd.concat([cuantitativas, cualitativas], axis=1)
    return(Datos_sin_nulos)

################################################ Funciones string concreto para sustitución de valores nulos ###################################################

def sust_string(dataframe,cadena):
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    cuantitativas = dataframe.select_dtypes(include=['float64', 'int64','float','int'])
    cualitativas_con_nulos = dataframe.select_dtypes(include=['object', 'datetime','category'])
    cualitativas = cualitativas_con_nulos.fillna(cadena)
    Datos_sin_nulos = pd.concat([cuantitativas, cualitativas], axis=1)
    return(Datos_sin_nulos)

################################################ Funciones constante para sustitución de valores nulos #########################################################

def sust_constante(dataframe,constante):
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    cuantitativas_con_nulos = dataframe.select_dtypes(include=['float64', 'int64','float','int'])
    cualitativas = dataframe.select_dtypes(include=['object', 'datetime','category'])
    cuantitativas = cuantitativas_con_nulos.fillna(constante)
    Datos_sin_nulos = pd.concat([cuantitativas, cualitativas], axis=1)
    return(Datos_sin_nulos)

################################################ Funciones median para sustitución de valores nulos #########################################################
 
def sust_mediana(dataframe):
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    cuantitativas_con_nulos = dataframe.select_dtypes(include=['float64', 'int64','float','int'])
    cualitativas = dataframe.select_dtypes(include=['object', 'datetime','category'])
    cuantitativas = cuantitativas_con_nulos.fillna(round(cuantitativas_con_nulos.median(), 1))
    Datos_sin_nulos = pd.concat([cuantitativas, cualitativas], axis=1)
    return(Datos_sin_nulos)

################################################ Funciones Identificación de valores nulos por columna y por dataframe ######################################

def contar_nulos(dataframe):
    import pandas as pd
    import numpy as np
    import matplotlib as plt
    nulos_columna = dataframe.isnull().sum()
    nulos_dataframe = dataframe.isnull().sum() .sum()
    print('Nulos por columna:', nulos_columna)
    print('Nulos por dataframe:', nulos_dataframe)
    return

################################################ Función Sustituye Valores Atípicos Método de “Rango intercuartílico” ######################################

def sust_atipicos(df):
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import diego as dg
    dfc = df.copy()
    dfc1 = dg.sust_ffill(dfc)
    dfc2 = dg.sust_prom(dfc1)
    cuantitativas = dfc2.select_dtypes(include=['int64', 'float64'])
    cualitativas = dfc2.select_dtypes(include=['object','datetime64[ns]'])
    y = cuantitativas
    percenttile25 = y.quantile(0.25)
    percenttile75 = y.quantile(0.75) 
    iqr = percenttile75 - percenttile25
    Limite_Superior_iqr = percenttile75 + 1.5 * iqr
    Limite_Inferior_iqr = percenttile25 - 1.5 * iqr
    iqr = cuantitativas[(y<Limite_Superior_iqr) & (y>Limite_Inferior_iqr)]
    iqr2 = iqr.copy()
    iqr2 = dg.sust_prom(iqr2)
    iqr2 = iqr2.dropna(axis=1, how='all')
    nulos_iqr2 = dg.contar_nulos(iqr2)
    Datos_limpios = pd.concat([cualitativas, iqr2], axis=1)
    return Datos_limpios

################################################ Función que aplica regla de Sturges ######################################

def categorizacion_sturges(df, columna):
    import pandas as pd
    import numpy as np
    from funpymodeling.exploratory import freq_tbl

    df_filtrado = df[df[columna] != 0]
    n = len(df_filtrado)
    Max = df_filtrado[columna].max()
    Min = df_filtrado[columna].min()
    R = Max - Min
    ni = 1 + 3.32 * np.log10(n)
    i = R / ni
    intervalos = np.linspace(Min, Max, num=15)
    categorias = [f"{intervalos[i]:.2f} - {intervalos[i+1]:.2f}" for i in range(len(intervalos)-1)]
    
    df_filtrado[columna] = pd.cut(
        x=df_filtrado[columna],
        bins=intervalos,
        labels=categorias,
        include_lowest=True
    )
    
    tabla = freq_tbl(df_filtrado[columna])
    tabla2 = tabla.drop(['percentage', 'cumulative_perc'], axis=1)
    filtro = tabla2[tabla2['frequency'] >= 1]
    filtro_index = filtro.set_index(columna)
    
    return filtro_index

################################################ Función que aplica regla de Sturges y devuelve las categorías ######################################

def categorias_sturges(df, columna):
    import pandas as pd
    import numpy as np
    from funpymodeling.exploratory import freq_tbl

    df_filtrado = df[df[columna] != 0]
    n = len(df_filtrado)
    Max = df_filtrado[columna].max()
    Min = df_filtrado[columna].min()
    R = Max - Min
    ni = 1 + 3.32 * np.log10(n)
    i = R / ni
    intervalos = np.linspace(Min, Max, num=15)
    categorias = [f"{intervalos[i]:.2f} - {intervalos[i+1]:.2f}" for i in range(len(intervalos)-1)]
    
    df_filtrado[columna] = pd.cut(
        x=df_filtrado[columna],
        bins=intervalos,
        labels=categorias,
        include_lowest=True
    )
    
    tabla = freq_tbl(df_filtrado[columna])
    tabla2 = tabla.drop(['percentage', 'cumulative_perc'], axis=1)
    filtro = tabla2[tabla2['frequency'] >= 1]
    filtro_index = filtro.set_index(columna)
    
    return categorias


