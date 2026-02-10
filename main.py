from flask import Flask 
from app import views

app = Flask(__name__)

app.add_url_rule('/', 'index', views.index)

app.add_url_rule('/app', 'app', views.app)

app.add_url_rule('/app/gender', 'genderapp', views.genderapp, methods = ['GET', 'POST'])

if __name__ == "__main__":
    app.run(debug=True)