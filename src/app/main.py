from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

colunas = ['tamanho','ano','garagem']
modelo = pickle.load(open('/Users/kaduangelucci/Documents/Estudos/Alura/Deploy de ML/ml_deploy/models/modelo.sav','rb'))

app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = os.environ.get("user_name")
# app.config['BASIC_AUTH_USERNAME'] = 'kadu'
app.config['BASIC_AUTH_PASSWORD'] = os.environ.get('user_password')
# app.config['BASIC_AUTH_PASSWORD'] = '32428283'

basic_auth = BasicAuth(app)

@app.route('/')
def home():
    return "Minha primeira API."

@app.route('/sentimento/<frase>')
@basic_auth.required
def sentimento(frase):
    tb = TextBlob(frase)
    # tb_en = tb.translate(to='en')
    polaridade = tb.sentiment.polarity
    return "polaridade: {}".format(polaridade)

@app.route('/cotacao/', methods=['POST'])
@basic_auth.required
def cotacao():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=preco[0])

app.run(debug=True, host='0.0.0.0', port=5002)
