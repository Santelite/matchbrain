# MatchBrain
MatchBrain es un algoritmo de predicción de resultado y cantidad de goles de partidos de futbol escrito en Python, utilizando LightGBM (Light Gradient-Boosting Machine) el cual es un aloritmo de Machine Learning basado en arboles de decisión. El modelo está entrenado con un dataset que incluyen todos los partidos internacionales de naciones jugados, desde amistosos hasta mundiales, pero el modelo ya se presta para entrenarse con datos de Clubes*.

*(quizás)

## Capacidades
MatchBrain actualmente realiza las siguientes predicciones:
- Ganador del partido
- Cantidad de goles totales
- Si el partido tendra 3 goles (OVER 2.5 GOALS)
- Marcador Exacto

Aparte de eso, calcula la forma actual ofensiva y defensiva, y les da a las Naciones un puntaje ELO calculado historico para todos los partidos que estén en el dataset.

### Validación con Partidos Reales (6 y 7 de junio de 2026)
 
Prueba realizada con partidos amistosos pre-mundialistas (El escenario más complicado para un modelo así):

| Partido | Marcador Real | Marcador Simulado | Prob. Resultado (Local/Empate/Visitante) | Goles Esperados | ¿Cumplió Over 2.5? | ¿Cumplió Resultado? | ¿Cumplió Marcador? |
|---|---|---|---|---|---|---|---|
| Ecuador vs Guatemala | 3 - 0 | 1 - 0 | 67% / 21% / 12% | 2.64 ✅ | ✅ Sí (3 goles) | ✅ Sí (L) | ❌ No |
| Greece vs Italy | 0 - 1 | 1 - 1 | 23% / 23% / 54% | 2.96 ✅ | ❌ No (1 gol) | ✅ Sí (V) | ❌ No |
| Colombia vs Jordan | 2 - 0 | 2 - 0 | 65% / 20% / 15% | 2.97 ✅ | ❌ No (2 goles) | ✅ Sí (L) | ✅ Sí |
| Denmark vs Ukraine | 2 - 1 | 1 - 0 | 56% / 23% / 21% | 2.79 ✅ | ✅ Sí (3 goles) | ✅ Sí (L) | ❌ No |
| Morocco vs Norway | 1 - 1 | 1 - 1 | 44% / 27% / 30% | 2.52 ✅ | ❌ No (2 goles) | ✅ Sí (E) | ✅ Sí |
| Kosovo vs Andorra | 3 - 0 | 3 - 0 | 89% / 8% / 3% | 3.62 ✅ | ✅ Sí (3 goles) | ✅ Sí (L) | ✅ Sí |
| Armenia vs Kazakhstan | 1 - 1 | 1 - 2 | 14% / 18% / 68% | 3.47 ✅ | ❌ Sí (2 goles) | ❌ No (E) | ❌ No |
| Estonia vs Faroe Islands | 1 - 0 | 1 - 1 | 37% / 28% / 35% | 2.41 ❌ | ❌ No (1 gol) | ✅ Sí (L) | ❌ No |
| Gibraltar vs Cayman Islands | 4 - 1 | 1 - 1 | 33% / 26% / 41% | 2.64 ✅ | ✅ Sí (5 goles) | ❌ No (L) | ❌ No |
| Portugal vs Chile | 2 - 1 | 1 - 0 | 60% / 22% / 18% | 2.73 ✅ | ✅ Sí (3 goles) | ✅ Sí (L) | ❌ No |
| Romania vs Wales | 2 - 1 | 1 - 1 | 29% / 27% / 44% | 2.42 ❌ | ✅ Sí (3 goles) | ❌ No (V) | ❌ No |
| Albania vs Luxembourg | 0 - 1 | 1 - 0 | 62% / 23% / 15% | 2.48 ❌ | ❌ No (1 gol) | ❌ No (V) | ❌ No |
| USA vs Germany | 1 - 2 | 1 - 1 | 23% / 22% / 55% | 3.18 ✅ | ✅ Sí (3 goles) | ✅ Sí (V) | ❌ No |
| Panama vs Bosnia and Herzegovina | 1 - 1 | 1 - 1 | 51% / 25% / 24% | 2.58 ✅ | ❌ No (2 goles) | ✅ Sí (E) | ✅ Sí |
| Switzerland vs Australia | 1 - 1 | 1 - 1 | 44% / 26% / 30% | 2.65 ✅ | ❌ No (2 goles) | ✅ Sí (E) | ✅ Sí |
| Bolivia vs Scotland | 0 - 4 | 1 - 1 | 29% / 27% / 43% | 2.42 ❌ | ✅ Sí (4 goles) | ✅ Sí (V) | ❌ No |
| England vs New Zealand | 1 - 0 | 2 - 0 | 75% / 17% / 8% | 2.80 ✅ | ❌ No (1 gol) | ✅ Sí (L) | ❌ No |
| Qatar vs El Salvador | 0 - 0 | 1 - 0 | 54% / 26% / 20% | 2.32 ❌ | ❌ No (0 goles) | ❌ No (E) | ❌ No |
| Brazil vs Egypt | 2 - 1 | 1 - 0 | 56% / 24% / 20% | 2.60 ✅ | ✅ Sí (3 goles) | ✅ Sí (L) | ❌ No |
| Argentina vs Honduras | 2 - 0 | 2 - 0 | 76% / 16% / 8% | 2.83 ✅ | ❌ No (2 goles) | ✅ Sí (L) | ✅ Sí |

---

### Resumen General

| Métrica | Aciertos | Total | % Acierto |
|---|---|---|---|
| Resultado (L/E/V) | 13 | 20 | **65%** |
| Marcador Exacto | 6 | 20 | **30%** |
| Over 2.5 goles | 11 | 20 | **55%** |
---

# Introducción
El Fútbol es un deporte caotico e impredecible, y eso lo hace particularmente interesante al verlo de un punto de vista analítico. Apesar del caos el fútbol es un deporte extremadamente analítico, es un deporte donde variables como el rendimiento histórico, las estadísticas de los jugadores, las condiciones del encuentro, las tácticas empleadas y otros factores contextuales pueden afectar significativamente el resultado de un partido. Como podrás imaginarte, esto produce muchos datos.

Debido a esto, se hace un deporte bastante ideal para ajustar a un modelo de Machine Learning, ya que ahora las computadoras son capaces de procesar millones de datos por segundo y encontrar patrones complejos en grandes cantidades de datos, cosa que a un humano puede tomarle mucho tiempo.

La idea con MatchBrain era combinar mi pasión por el fútbol con mi fascinación con la Ciencia de Datos y el Machine Learning. Mi objetivo final era aprender más sobre la parte analítca del fútbol y encontrar patrones que nunca se me habrían ocurrido por mi cuenta. Ningún modelo es perfecto, pero creo que he logrado hacer algo util, bastante liviano, y descubriendo algún que otro dato interesante.

# Como funciona?

## Dataset
El dataset (o dataframe) inicial contiene las filas:
- `date`
- `home_team`
- `away_team`
- `home_score`
- `away_score`
- `tournament`
- `city`
- `country` (país en el que se jugó el encuentro)
- `neutral`

## El Algoritmo

El sistema utiliza LightGBM porque es de lo mejorcito que existe actualmente para un dataset estructurado como este, es muy rápido y consume mucha menos memoria que otros algoritmos de Gradient Boosting. Los modelos de Gradient Boosting construyen muchos árboles pequeños que corrigen los errores de los anteriores, permitiendo capturar patrones difíciles sin diseñarlos manualmente.

## Feature Engineering

Uno de los mayores problemas del fútbol es que tiene una complicación matemática muy molesta en comparación con otros datos temporales, me explico: un equipo puede jugar hoy como `home_team` y el próximo partido como `away_team`, puede jugar mal por que a un defensa le dio dolor de estomago, o al delantero estrella se le olvido como jugar futbol y ganar mañana por goleada. No se puede simplemente agrupar por una sola columna, tenes que calcular el rendimiento histórico "global" de cada selección, sin importar si jugó en casa o de visita, y luego asociar ese pasado a cada fila del dataset.

El sistema predice a partir de cuatro dimensiones:
- La forma reciente de los equipos. (Como esta actualmente)
- El contexto (Que se juega, y donde se juega)
- El potencial de ataque de ambos equipos comparado la defensa del otro.
- El puntaje ELO Historico. (Como se comporta historicamente)

Entonces la mejor forma de comenzar es haciendo dos copias de los datasets, y dos modelos de predicción, uno para el que juega en casa y otro para el que juega de visita, y entrenarlos individualmente.

### Forma actual

Para comenzar había que capturar la "forma reciente" de cada selección, se duplica el dataset en dos perspectivas (local y visitante), se crea un historial cronológico por equipo y se aplica `.shift(1).rolling(10).mean()` para evitar filtración de datos futuros (para que el modelo no haga trampa solo viendo el resultado real del partido que debe predecir.).
 
| Feature | Descripción |
|---|---|
| `goles_anotados_form_home` | Promedio de goles anotados por el local en sus últimos 10 partidos |
| `goles_recibidos_form_home` | Promedio de goles recibidos por el local en sus últimos 10 partidos |
| `goles_anotados_form_away` | Promedio de goles anotados por el visitante en sus últimos 10 partidos |
| `goles_recibidos_form_away` | Promedio de goles recibidos por el visitante en sus últimos 10 partidos |

### Contexto
El contexto en el que se juega el partido.

| Feature | Descripción |
|---|---|
| `is_home_advantage` | Binaria (1 si no es cancha neutral, 0 si es neutral) |
| `tipo_torneo` | Categoría del torneo: `mundial`, `oficial_continental`, `amistoso`, `otros_torneos` |

Como en el dataset vienen muchos torneos que nadie conoce, había que clasificarlos y ponerlos a niveles parejos sin importar el continente:

| Clasificación | Torneos |
|---|---|
| `mundial` | Copa Mundial de la FIFA |
| `oficial_continental` | Partido oficial que tenga en juego un titulo o clasificación (EURO, Copa América, Qualifiers para la Copa Mundial, etc.) |
| `amistoso` | Partidos amistosos, Nations League, etc. |

### Potencial de Ataque

Ratios que combinan el ataque de un equipo con la defensa del rival, facilitando el aprendizaje del modelo:
 
```python
df['potencial_ataque_home'] = df['goles_anotados_form_home'] / (df['goles_recibidos_form_away'] + 0.1)
df['potencial_ataque_away'] = df['goles_anotados_form_away'] / (df['goles_recibidos_form_home'] + 0.1)
```

### Sistema de ELO
**Este es el mismo sistema utilizado oficialmente por el ajedrez en sitios como chess.com, y se aproxima bastante al Ranking FIFA Oficial que estaba roto desde el inicio, hasta que decidieron arreglarlo ahí por 2018. [Al menos tiene la misma base matemática](https://en.wikipedia.org/wiki/Elo_rating_system#Theory).**

Todos los países comienzan con un rating de referencia:
 
`1500`
 
Este valor es arbitrario pero comodo porque evita números negativos, ofrece rango suficiente por encima y por debajo, y el sistema es invariante a la escala (funcionaría igual con 0 o con 5000). Al ser un **sistema de suma cero**, los puntos no se crean ni se destruyen, justo como la materia, todo punto que gana un equipo se lo resta al rival y eso depende de que tan disparejo sea el partido *(No es lo mismo ganarle a Brazil en un mundial, que ganarle a El Salvador en un amistoso en cualquier momento de su historia).*

Antes de cada partido, el sistema calcula la probabilidad esperada de que el Equipo A gane, basándose en la diferencia de ratings actuales:
 
$$E_A = \frac{1}{1 + 10^{\,(R_B - R_A)\,/\,400}}$$
 
$$E_B = 1 - E_A$$
 
El divisor **400** es el parámetro de escala de la curva logística. Su significado concreto es:
 
> Una diferencia de exactamente 400 puntos de ELO implica que el favorito tiene aproximadamente un **91% de probabilidad de ganar**.
 
Demostración numérica: si $R_A = 1900$ y $R_B = 1500$:
 
$$E_A = \frac{1}{1 + 10^{(1500-1900)/400}} = \frac{1}{1 + 10^{-1}} = \frac{1}{1.1} \approx 0.909$$

Una vez conocido el resultado real, ambos ratings se actualizan simultáneamente:
 
$$R_A' = R_A + K \cdot (S_A - E_A)$$
 
$$R_B' = R_B + K \cdot (S_B - E_B)$$
 
Donde $S$ es el resultado real codificado:
 
| Resultado | $S_A$ | $S_B$ |
|---|---|---|
| Gana el Equipo A | 1.0 | 0.0 |
| Empate | 0.5 | 0.5 |
| Gana el Equipo B | 0.0 | 1.0 |
 
El término $(S_A - E_A)$ representa la **sorpresa del resultado**, osea que tan inesperado era el resultado.

Tenemos que:
- Si el favorito ($E_A = 0.91$) gana como se esperaba: $(1 - 0.91) = 0.09$ hay un ajuste mínimo.

PERO:
- Si el débil ($E_A = 0.09$) da la sorpresa y gana: $(1 - 0.09) = 0.91$ al equipo débil le da una cantidad de puntos masiva.

$K$ actúa como la **tasa de aprendizaje** que controla cuántos puntos máximos se pueden transferir en un solo partido y aqui se usa un $K$ variable según la importancia del torneo:
 
$$K = \begin{cases} 60 & \text{si el partido es de Copa del Mundo} \\ 
40 & \text{si es torneo oficial continental o clasificatoria} \\ 
20 & \text{si es amistoso u otro torneo menor} \end{cases}$$

En este sentido $K$ es la importancia que tiene el partido
 
Esto es igual que el estándar adoptado por la **FIFA en 2018** (sistema SUM) despues que se dieran cuenta que su sistema actual era una basura. Este reconoce que una derrota en una final del mundo debe pesar más que una derrota en un partido de preparación. El ELO es matemáticamente inmune al problema del viejo Ranking FIFA (inflar el rating jugando muchos amistosos fáciles).

Por ejemplo: si un equipo fuerte ($R_A = 1800$) juega contra un rival débil ($R_B = 1100$)
 
$$E_A = \frac{1}{1 + 10^{(1100-1800)/400}} \approx 0.986$$
 
Si gana (lo esperado): $\Delta R_A = 20 \times (1 - 0.986) = +0.28$ puntos. Casi nada.
 
Si **empata** (resultado inesperado): $\Delta R_A = 20 \times (0.5 - 0.986) = -9.7$ puntos. Castigo severo por no ganar lo que se suponía debía ganar.

El cálculo es estrictamente secuencial (cada partido depende del anterior), por lo que se implementa con un bucle iterativo sobre el dataframe ordenado cronológicamente. El ELO se le da al modelo a la hora de hacer una predicción es **el rating vigente justo antes de que se dispute el partido**

En codigo todo esto se ve así:
```python
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
```

## Que se encontró:
<img width="890" height="625" alt="away" src="https://github.com/user-attachments/assets/de3e8db9-1917-4473-9c60-f487d08cd445" />
<img width="890" height="625" alt="home" src="https://github.com/user-attachments/assets/46e248f8-d480-4272-8997-78cfcdac9b69" />

Curiosamente, aparte del ELO propio, el modelo encontró que importa más que tan débil es la defensa del rival, sobre que tan poderosa es la ofensa propia, de igual forma tienen un poco de influencia los factores del contexto (Terreno neutral, Tipo del torneo)

## Límites 
La arquitectura de dos modelos separados funciona excepcionalmente bien pero asume una regla que fútbol real viola constantemente: que los goles del equipo local y los del visitante son **eventos independientes**, En la práctica, un gol en el minuto 5 cambia por completo la postura táctica de ambos equipos. Aún así, el modelo ha probado ser bastante acertado y tiene suficientes datos para producir buenas predicciones.

## Desplegar por tu cuenta
Se requiere Python 3.12+ preferiblemente con venv. Despues de clonar el repositorio (asumiendo que estes en la carpeta raiz) puedes realizar:
```python
/.venv/Scripts/activate
pip install -r requirements.txt
python app.py
```


## Preguntas Frecuentes
### Si pero porque?
Estaba aburrido.

## Planes
Crear un frontend en html y adaptar el modelo a una API REST para desplegar una pagina web.
