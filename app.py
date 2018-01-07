from flask import Flask, jsonify
from flask import render_template

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def get_tasks():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
