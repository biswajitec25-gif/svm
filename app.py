import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template, request, send_file
import os
from learning_content.fundamentals import FUNDAMENTALS
from learning_content.kernels import KERNELS
from learning_content.parameters import PARAMETERS
from models.svm_model import generate_linear_data, generate_moon_data, generate_circular_data, train_svm, plot_decision_boundary

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/learning/fundamentals')
def learning_fundamentals():
    return render_template('learning/fundamentals.html', content=FUNDAMENTALS)

@app.route('/learning/kernels')
def learning_kernels():
    return render_template('learning/kernels.html', content=KERNELS)

@app.route('/learning/parameters')
def learning_parameters():
    return render_template('learning/parameters.html', content=PARAMETERS)

@app.route('/python_prog')
def python_prog():
    return render_template('python_prog/index.html')

@app.route('/lab')
def lab():
    return render_template('lab/index.html')

@app.route('/lab/linear')
def lab_linear():
    X, y = generate_linear_data()
    clf = train_svm(X, y, kernel='linear')
    plot_path = os.path.join('static', 'images', 'plots', 'linear.png')
    plot_decision_boundary(clf, X, y, plot_path)
    return render_template('lab/linear.html', plot=plot_path)

@app.route('/lab/nonlinear')
def lab_nonlinear():
    X, y = generate_moon_data()
    clf = train_svm(X, y, kernel='rbf')
    plot_path = os.path.join('static', 'images', 'plots', 'nonlinear.png')
    plot_decision_boundary(clf, X, y, plot_path)
    return render_template('lab/nonlinear.html', plot=plot_path)

@app.route('/csv_train')
def csv_train():
    return render_template('csv_train/index.html')

@app.route('/evaluation')
def evaluation():
    return render_template('evaluation/index.html')

@app.route('/quiz')
def quiz():
    return render_template('quiz/index.html')

@app.route('/interview')
def interview():
    return render_template('interview/index.html')

if __name__ == '__main__':
    app.run(debug=True)
