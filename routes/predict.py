from flask import Blueprint, request, jsonify
import joblib
import pandas as pd
import numpy as np
from scipy.stats import poisson

predecir = Blueprint('predict', __name__)

model_home = joblib.load('model_home.pkl')
model_away = joblib.load('model_away.pkl')
estado_paises = pd.read_pickle('estado_actual_paises.pkl')

@predecir.route('/elo', methods=['GET'])
def elo():
    elo_data = estado_paises.drop(
        columns=['date', 'goles_anotados_form', 'goles_recibidos_form', 'potencial_ataque']
    ).sort_values(by='elo', ascending=False).to_dict()
    return jsonify(elo_data)

@predecir.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    pais_home = data.get('home_team')
    pais_away = data.get('away_team')
    torneo = data.get('tipo_torneo', 'amistoso')
    es_neutral = data.get('neutral', False)
    
    if pais_home not in estado_paises.index or pais_away not in estado_paises.index:
        return jsonify({'error': 'Uno o ambos países no existen en el dataset'}), 400
        
    racha_home = estado_paises.loc[pais_home]
    racha_away = estado_paises.loc[pais_away]
    
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
    
    lambda_home = model_home.predict(datos_partido)[0]
    lambda_away = model_away.predict(datos_partido)[0]
    
    max_goles = 6
    matriz_goles = np.outer(poisson.pmf(range(max_goles), lambda_home),
                            poisson.pmf(range(max_goles), lambda_away))
    
    prob_empate = float(np.sum(np.diag(matriz_goles)))
    prob_home = float(np.sum(np.tril(matriz_goles, -1)))
    prob_away = float(np.sum(np.triu(matriz_goles, 1)))
    marcador_mas_probable = [int(x) for x in np.unravel_index(np.argmax(matriz_goles), matriz_goles.shape)]
    
    return jsonify({
        'partido': f"{pais_home} vs {pais_away}",
        'elo' : {
            'local': racha_home['elo'],
            'visitante': racha_away['elo']
        },
        'expectativa_goles': {
            'local': round(lambda_home, 2),
            'visitante': round(lambda_away, 2),
            'totales_esperados': round(lambda_home + lambda_away, 2)
        },
        'probabilidades_1X2': {
            'local': f"{prob_home*100:.2f}%",
            'empate': f"{prob_empate*100:.2f}%",
            'visitante': f"{prob_away*100:.2f}%"
        },
        'marcador_mas_probable': f"{marcador_mas_probable[0]} - {marcador_mas_probable[1]}",
        'formas': {
            'local': {
                'ofensiva': racha_home['goles_anotados_form'],
                'defensiva': racha_home['goles_recibidos_form'],
                'potencial_ataque': racha_home['potencial_ataque']
            },
            'visitante': {
                'ofensiva': racha_away['goles_anotados_form'],
                'defensiva': racha_away['goles_recibidos_form'],
                'potencial_ataque': racha_away['potencial_ataque']
            }
        }
    })