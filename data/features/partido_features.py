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

    
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    #separación de los datasets
    df_home = df[['date', 'home_team', 'home_score', 'away_score']].copy()
    df_home.columns = ['date', 'team', 'goles_anotados', 'goles_recibidos']
    df_home['was_home'] = 1

    
    df_away = df[['date', 'away_team', 'away_score', 'home_score']].copy()
    df_away.columns = ['date', 'team', 'goles_anotados', 'goles_recibidos']
    df_away['was_home'] = 0

    
    historial_equipos = pd.concat([df_home, df_away]).sort_values(['team', 'date']).reset_index(drop=True)
    
    N_PARTIDOS = 10

    #calculo de formas recientes

    historial_equipos['form_goles_anotados'] = (
        historial_equipos.groupby('team')['goles_anotados']
        .transform(lambda x: x.shift(1).rolling(N_PARTIDOS, min_periods=1).mean())
    )

    historial_equipos['form_goles_recibidos'] = (
        historial_equipos.groupby('team')['goles_recibidos']
        .transform(lambda x: x.shift(1).rolling(N_PARTIDOS, min_periods=1).mean())
    )

    
    
    historial_equipos['form_goles_anotados'] = historial_equipos['form_goles_anotados'].fillna(0)
    historial_equipos['form_goles_recibidos'] = historial_equipos['form_goles_recibidos'].fillna(0)

    
    features_home = historial_equipos[historial_equipos['was_home'] == 1][['date', 'team', 'form_goles_anotados', 'form_goles_recibidos']]
    features_home.columns = ['date', 'home_team', 'goles_anotados_form_home', 'goles_recibidos_form_home']

    features_away = historial_equipos[historial_equipos['was_home'] == 0][['date', 'team', 'form_goles_anotados', 'form_goles_recibidos']]
    features_away.columns = ['date', 'away_team', 'goles_anotados_form_away', 'goles_recibidos_form_away']

    
    df_final = pd.merge(df, features_home, on=['date', 'home_team'], how='left')
    df_final = pd.merge(df_final, features_away, on=['date', 'away_team'], how='left')

    
    
    df_final['is_home_advantage'] = np.where(df_final['neutral'] == False, 1, 0)

    df_final['tipo_torneo'] = df_final['tournament'].apply(mapear_torneo)

    
    df_final['potencial_ataque_home'] = df_final['goles_anotados_form_home'] / (df_final['goles_recibidos_form_away'] + 0.1)
    
    df_final['potencial_ataque_away'] = df_final['goles_anotados_form_away'] / (df_final['goles_recibidos_form_home'] + 0.1)

    print("Cálculo de ELO")
    
    #calculo de elo
    ELO_INICIAL = 1500
    k = 30

    elo_actual = {}

    elo_home_antes = []
    elo_away_antes = []

    for idx, row in df_final.iterrows():
        home = row['home_team']
        away = row['away_team']

        
        if row['tipo_torneo'] == 'mundial':
            K = 60  
        elif row['tipo_torneo'] == 'oficial_continental':
            K = 40  
        else:
            K = 20  
        
        
        r_home = elo_actual.get(home, ELO_INICIAL)
        r_away = elo_actual.get(away, ELO_INICIAL)
        
        
        elo_home_antes.append(r_home)
        elo_away_antes.append(r_away)
        
        
        e_home = 1 / (1 + 10 ** ((r_away - r_home) / 400))
        e_away = 1 / (1 + 10 ** ((r_home - r_away) / 400))
        
        
        if row['home_score'] > row['away_score']:
            s_home, s_away = 1.0, 0.0  
        elif row['home_score'] < row['away_score']:
            s_home, s_away = 0.0, 1.0  
        else:
            s_home, s_away = 0.5, 0.5  
            
        
        elo_actual[home] = r_home + k * (s_home - e_home)
        elo_actual[away] = r_away + k * (s_away - e_away)
    
    
    df_final['elo_home'] = elo_home_antes
    df_final['elo_away'] = elo_away_antes

    print("Dataframe terminado")
    
    #guardar el dataframe y procesarlo para tener los ultimos datos de las selecciones.
    df_final = df_final[df_final['date'].between('2000-01-01', '2026-06-04')]

    df_home = df_final[['date', 'home_team', 'goles_anotados_form_home', 'goles_recibidos_form_home','potencial_ataque_home', 'elo_home']].copy()
    df_home.columns = ['date', 'team', 'goles_anotados_form', 'goles_recibidos_form', 'potencial_ataque', 'elo']

    df_away = df_final[['date', 'away_team', 'goles_anotados_form_away', 'goles_recibidos_form_away','potencial_ataque_away', 'elo_away']].copy()
    df_away.columns = ['date', 'team', 'goles_anotados_form', 'goles_recibidos_form', 'potencial_ataque', 'elo']

    historial_completo = pd.concat([df_home, df_away]).sort_values('date')

    ultimo_partido = historial_completo.drop_duplicates(subset=['team'], keep='last').set_index('team')
    ultimo_partido.to_csv("estado_actual.csv")

    return df_final


def get_historial():
    return ultimo_partido