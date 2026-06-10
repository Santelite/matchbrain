from flask import Flask
from flask_cors import CORS
from routes.predict import predecir
from routes.main import main

app = Flask(__name__)
CORS(app) # Enables Cross-Origin requests and handles OPTIONS preflights automatically

app.register_blueprint(predecir, url_prefix='/api')
app.register_blueprint(main)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
