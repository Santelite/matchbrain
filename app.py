import pandas as pd
from services.resultado import predecir_partido, entrenar_modelo_resultado

# Entrenar Modelos
model_home, model_away = entrenar_modelo_resultado()
# ¡A adivinar!
predecir_partido(model_home, model_away, 'United States', 'Germany', es_neutral=True, torneo='mundial')