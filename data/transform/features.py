import pandas as pd
import numpy as np

ultimo_partido = "global"

def mapear_torneo(torneo):
    torneo = torneo.lower()
    if 'world cup' in torneo and 'qualifying' not in torneo:
        return 'mundial'
    elif 'qualification' in torneo or 'euro' in torneo or 'copa américa' in torneo or 'nations league' in torneo:
        return 'oficial_continental'
    elif 'friendly' in torneo:
        return 'amistoso'
    else:
        return 'otros_torneos'

def construir_dataset_features():
    global ultimo_partido
    df = pd.read_csv('./data/results.csv')

    # Convertir la fecha a formato datetime y ordenar cronológicamente (CRUCIAL)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    # 1. Perspectiva Local
    df_home = df[['date', 'home_team', 'home_score', 'away_score']].copy()
    df_home.columns = ['date', 'team', 'goles_anotados', 'goles_recibidos']
    df_home['was_home'] = 1

    # 2. Perspectiva Visitante
    df_away = df[['date', 'away_team', 'away_score', 'home_score']].copy()
    df_away.columns = ['date', 'team', 'goles_anotados', 'goles_recibidos']
    df_away['was_home'] = 0

    # Unir ambos en un solo gran dataframe de "rendimiento por partido"
    historial_equipos = pd.concat([df_home, df_away]).sort_values(['team', 'date']).reset_index(drop=True)

    print("Rolling...")
    # Definimos cuántos partidos hacia atrás queremos mirar (ej. últimos 5 partidos)
    N_PARTIDOS = 10

    # Agrupamos por equipo y calculamos las medias móviles desplazadas un lugar
    historial_equipos['form_goles_anotados'] = (
        historial_equipos.groupby('team')['goles_anotados']
        .transform(lambda x: x.shift(1).rolling(N_PARTIDOS, min_periods=1).mean())
    )

    historial_equipos['form_goles_recibidos'] = (
        historial_equipos.groupby('team')['goles_recibidos']
        .transform(lambda x: x.shift(1).rolling(N_PARTIDOS, min_periods=1).mean())
    )

    # Rellenar los primeros partidos de la historia (donde no hay historial) con 0 o la media global
    historial_equipos['form_goles_anotados'] = historial_equipos['form_goles_anotados'].fillna(0)
    historial_equipos['form_goles_recibidos'] = historial_equipos['form_goles_recibidos'].fillna(0)

    # Separar de nuevo lo que calculamos en local y visitante
    features_home = historial_equipos[historial_equipos['was_home'] == 1][['date', 'team', 'form_goles_anotados', 'form_goles_recibidos']]
    features_home.columns = ['date', 'home_team', 'goles_anotados_form_home', 'goles_recibidos_form_home']

    features_away = historial_equipos[historial_equipos['was_home'] == 0][['date', 'team', 'form_goles_anotados', 'form_goles_recibidos']]
    features_away.columns = ['date', 'away_team', 'goles_anotados_form_away', 'goles_recibidos_form_away']

    # Hacer el "Merge" con el dataframe original de partidos
    df_final = pd.merge(df, features_home, on=['date', 'home_team'], how='left')
    df_final = pd.merge(df_final, features_away, on=['date', 'away_team'], how='left')

    # Crear la feature de ventaja de localía basándonos en la columna 'neutral'
    # Si no es neutral (False), el local tiene ventaja (1). Si es neutral (True), no (0).
    df_final['is_home_advantage'] = np.where(df_final['neutral'] == False, 1, 0)

    df_final['tipo_torneo'] = df_final['tournament'].apply(mapear_torneo)

    # Mide el potencial de ataque neto del Home frente a la defensa del Away
    df_final['potencial_ataque_home'] = df_final['goles_anotados_form_home'] / (df_final['goles_recibidos_form_away'] + 0.1)
    # Mide el potencial de ataque neto del Away frente a la defensa del Home
    df_final['potencial_ataque_away'] = df_final['goles_anotados_form_away'] / (df_final['goles_recibidos_form_home'] + 0.1)

    print("Cálculo de ELO")
    # Calculo de Sistema ELO
    ELO_INICIAL = 1500
    FACTOR_K = 30

    elo_actual = {}

    elo_home_antes = []
    elo_away_antes = []

    for idx, row in df_final.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        # Si es el primer partido de la historia de ese país, le asignamos el Elo inicial
        r_home = elo_actual.get(home, ELO_INICIAL)
        r_away = elo_actual.get(away, ELO_INICIAL)
        
        # Guardamos el Elo actual en nuestras listas de características antes de que cambie
        elo_home_antes.append(r_home)
        elo_away_antes.append(r_away)
        
        # ---- CALCULAR EXPECTATIVAS (Paso A) ----
        e_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        e_away = 1 / (1 + 10 ** ((r_home - r_away) / 400))
        
        # ---- DETERMINAR RESULTADO REAL (Paso B) ----
        if row['home_score'] > row['away_score']:
            s_home, s_away = 1.0, 0.0  # Gana Local
        elif row['home_score'] < row['away_score']:
            s_home, s_away = 0.0, 1.0  # Gana Visitante
        else:
            s_home, s_away = 0.5, 0.5  # Empate
            
        # ---- ACTUALIZAR EL DICCIONARIO VIVO ----
        elo_actual[home] = r_home + FACTOR_K * (s_home - e_home)
        elo_actual[away] = r_away + FACTOR_K * (s_away - e_away)
    
    # 4. Inyectar las nuevas columnas de características al DataFrame original
    df_final['elo_home'] = elo_home_antes
    df_final['elo_away'] = elo_away_antes

    print("Dataframe terminado")
    #Filtrar el dataframe final por fecha.
    df_final = df_final[df_final['date'].between('2000-01-01', '2026-06-04')]

    # 1. Volvemos a usar el truco de descomponer en un histórico por equipo
    df_home = df_final[['date', 'home_team', 'goles_anotados_form_home', 'goles_recibidos_form_home','potencial_ataque_home', 'elo_home']].copy()
    df_home.columns = ['date', 'team', 'goles_anotados_form', 'goles_recibidos_form', 'potencial_ataque', 'elo']

    df_away = df_final[['date', 'away_team', 'goles_anotados_form_away', 'goles_recibidos_form_away','potencial_ataque_away', 'elo_away']].copy()
    df_away.columns = ['date', 'team', 'goles_anotados_form', 'goles_recibidos_form', 'potencial_ataque', 'elo']

    # Unimos todo el histórico real y lo ordenamos por fecha
    historial_completo = pd.concat([df_home, df_away]).sort_values('date')

    # 2. LA MAGIA: Nos quedamos ÚNICAMENTE con la última fila de cada país
    # Al usar drop_duplicates con keep='last', conservamos solo el estado más reciente de la selección
    ultimo_partido = historial_completo.drop_duplicates(subset=['team'], keep='last').set_index('team')

    return df_final


def get_historial():
    return ultimo_partido