import pandas as pd
from services.predict_resultado import predecir_partido, entrenar_modelo_resultado

# Entrenar Modelos
model_home, model_away = entrenar_modelo_resultado()

predecir_partido(model_home, model_away, 'Ecuador', 'Guatemala', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Greece', 'Italy', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Colombia', 'Jordan', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Denmark', 'Ukraine', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Morocco', 'Norway', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Kosovo', 'Andorra', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Armenia', 'Kazakhstan', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Estonia', 'Faroe Islands', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Gibraltar', 'Cayman Islands', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Portugal', 'Chile', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Romania', 'Wales', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Albania', 'Luxembourg', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'United States', 'Germany', es_neutral=False, torneo='amistoso')
predecir_partido(model_home, model_away, 'Panama', 'Bosnia and Herzegovina', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Switzerland', 'Australia', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Bolivia', 'Scotland', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'England', 'New Zealand', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Qatar', 'El Salvador', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Brazil', 'Egypt', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Turkiye', 'Venezuela', es_neutral=True, torneo='amistoso')
predecir_partido(model_home, model_away, 'Argentina', 'Honduras', es_neutral=True, torneo='amistoso')
