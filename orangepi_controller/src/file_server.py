import os
import threading
from flask import Flask,send_from_directory

app = Flask(__name__)

app.config['STATIC_FOLDER'] = os.getcwd()

@app.route('/<random_num>/<filename>')
def uploaded_file(random_num,filename):
    return send_from_directory(app.config['STATIC_FOLDER'],filename)


app.run(host="0.0.0.0"
    , port=8010
    , debug=True)