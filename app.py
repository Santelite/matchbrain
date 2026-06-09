from flask import Flask
from routes.predict import predecir

app = Flask(__name__)

app.register_blueprint(predecir, url_prefix='/api')

@app.route('/')
def index():
    return "Endpoint activo /api/predict"

if __name__ == '__main__':
    app.run(debug=True, port=5000)
