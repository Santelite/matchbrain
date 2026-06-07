import lightgbm as lgb
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import poisson
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from data.transform.features import construir_dataset_features, get_historial

df_final = construir_dataset_features()

def entrenar_modelo_resultado():
    # --- CONFIGURACIÓN DE CARACTERÍSTICAS ---
    features = [
        'goles_anotados_form_home', 'goles_recibidos_form_home',
        'goles_anotados_form_away', 'goles_recibidos_form_away',
        'potencial_ataque_home', 'potencial_ataque_away',
        'elo_home', 'elo_away',
        'is_home_advantage', 'tipo_torneo'
    ]

    # Asegurar que las variables categóricas estén listas
    df_final['tipo_torneo'] = df_final['tipo_torneo'].astype('category')

    # Filtrar datos del año 2000 en adelante para evitar fútbol antiguo
    df_moderno = df_final[df_final['date'].dt.year >= 2000].copy()

    X = df_moderno[features]
    y_home = df_moderno['home_score']
    y_away = df_moderno['away_score']

    # Hiperparámetros regulados que encontramos juntos
    params = {
        'objective': 'poisson',  
        'metric': 'rmse',              
        'boosting_type': 'gbdt',       
        'learning_rate': 0.005,
        'num_boost_round': 2000,
        'num_leaves': 15,              
        'min_data_in_leaf': 50,       
        'lambda_l1': 1.5,              
        'lambda_l2': 1.5,              
        'feature_fraction': 0.6,
        'verbose': -1
    }

    # --- ENTRENAMIENTO MODELO HOME ---
    print("Entrenando Modelo Home Goles...")
    X_train_h, X_val_h, y_train_h, y_val_h = train_test_split(X, y_home, test_size=0.2, random_state=42)
    train_data_h = lgb.Dataset(X_train_h, label=y_train_h)
    val_data_h = lgb.Dataset(X_val_h, label=y_val_h, reference=train_data_h)

    model_home = lgb.train(
        params, train_data_h, num_boost_round=1000,
        valid_sets=[train_data_h, val_data_h],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=100)
                   ]
    )

    y_pred_h = model_home.predict(X_val_h)
    rmse_home = np.sqrt(mean_squared_error(y_val_h, y_pred_h))
    print(f"\nEl RMSE del modelo Home es: {rmse_home:.4f} goles.")

    # --- ENTRENAMIENTO MODELO AWAY ---
    print("\nEntrenando Modelo Away Goles...")
    X_train_a, X_val_a, y_train_a, y_val_a = train_test_split(X, y_away, test_size=0.2, random_state=42)
    train_data_a = lgb.Dataset(X_train_a, label=y_train_a)
    val_data_a = lgb.Dataset(X_val_a, label=y_val_a, reference=train_data_a)

    model_away = lgb.train(
        params, train_data_a, num_boost_round=1000,
        valid_sets=[train_data_a, val_data_a],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=100)
                   ]
    )

    y_pred_a = model_away.predict(X_val_a)
    rmse_away = np.sqrt(mean_squared_error(y_val_a, y_pred_a))
    print(f"El RMSE del modelo Away es: {rmse_away:.4f} goles.")

    lgb.plot_importance(model_home, max_num_features=10, importance_type='gain')
    plt.title("¿Qué factores influyen más en los goles locales? (HOME)")
    plt.savefig("home.png")

    lgb.plot_importance(model_away, max_num_features=10, importance_type='gain')
    plt.title("¿Qué factores influyen más en los goles locales? (AWAY)")
    plt.savefig("away.png")

    return model_home, model_away


def predecir_partido (model_home, model_away, pais_home, pais_away, es_neutral=False, torneo='amistoso'):
    estado_actual_paises = get_historial()
    if pais_home not in estado_actual_paises.index or pais_away not in estado_actual_paises.index:
        print("Uno de los países no se encuentra en el dataset histórico.")
        return
    
    # 2. Extraer de forma automática los últimos datos de racha para cada uno
    racha_home = estado_actual_paises.loc[pais_home]
    racha_away = estado_actual_paises.loc[pais_away]
    
    # 3. Construir el DataFrame con la estructura exacta que piden los modelos
    datos_partido = pd.DataFrame([{
        'goles_anotados_form_home': racha_home['goles_anotados_form'],
        'goles_recibidos_form_home': racha_home['goles_recibidos_form'],
        'goles_anotados_form_away': racha_away['goles_anotados_form'],
        'goles_recibidos_form_away': racha_away['goles_recibidos_form'],
        'potencial_ataque_home' : racha_home['potencial_ataque'],
        'potencial_ataque_away' : racha_away['potencial_ataque'],
        'elo_home' : racha_home['elo'],
        'elo_away' : racha_away['elo'],
        'is_home_advantage': 0 if es_neutral else 1,
        'tipo_torneo': torneo
    }])

    datos_partido['tipo_torneo'] = datos_partido['tipo_torneo'].astype('category')

    # 1. Obtener el lambda (goles esperados) de cada modelo
    lambda_home = model_home.predict(datos_partido)[0]
    lambda_away = model_away.predict(datos_partido)[0]
    
    lambda_total = lambda_away + lambda_home

    print(f"Simulación: {pais_home} vs {pais_away}\n\n")

    print(f"=== EXPECTATIVA DE GOLES ===")
    print(f"Goles esperados {pais_home}: {lambda_home:.2f}")
    print(f"Goles esperados {pais_away}: {lambda_away:.2f}\n")
    print(f"Goles esperados totales: {lambda_total:.2f}\n")
    if (lambda_total > 2.5): print("Se esperan más de 2.5 goles\n")

    # 2. Calcular matriz de probabilidades de marcadores (hasta 10 goles por equipo)
    max_goles = 11
    matriz_goles = np.outer(poisson.pmf(range(max_goles), lambda_home),
                            poisson.pmf(range(max_goles), lambda_away))
    
    # 3. Calcular probabilidades agregadas (1, X, 2)
    prob_empate = np.sum(np.diag(matriz_goles))
    prob_home = np.sum(np.tril(matriz_goles, -1))
    prob_away = np.sum(np.triu(matriz_goles, 1))
    
    # 4. Encontrar el marcador exacto más probable
    marcador_mas_probable = np.unravel_index(np.argmax(matriz_goles), matriz_goles.shape)
    
    print(f"=== PROBABILIDADES DEL RESULTADO ===")
    print(f"Victoria {pais_home}: {prob_home*100:.2f}%")
    print(f"Empate: {prob_empate*100:.2f}%")
    print(f"Victoria {pais_away}: {prob_away*100:.2f}%")
    print(f"Marcador más probable: {marcador_mas_probable[0]} - {marcador_mas_probable[1]}")