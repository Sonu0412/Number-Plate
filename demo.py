import numpy as np
from flask import Flask
import calendar as cla

app = Flask(__name__)
@app.route('/')
def home():
    c = cla.month(2025,5)
    a = np.random.randint(1,10)
    return f'<pre>{c}</pre>'

if __name__ == '__main__':
    app.run(debug=True)
