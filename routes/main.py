from flask import Blueprint, render_template, request


main = Blueprint('main',__name__)

@main.route('/')
def index():
    return render_template('main.html')

        
