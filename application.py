from flask import Flask,render_template,request
from sklearn import svm,datasets,metrics
from sklearn.model_selection import train_test_split
import sqlite3

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('test.html',
    features=list(datasets.load_iris()['feature_names']))

@app.route('/handle_data', methods=['POST'])
def handle_data():
    best_model = svm.SVC(kernel=request.form['kernel'],gamma=float(request.form['gamma']),
        C=float(request.form['c']),degree=int(request.form['poly']))
    X = datasets.load_iris().data[:,[int(request.form['feature1']),int(request.form['feature2'])]]
    Y = datasets.load_iris().target
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.4)
    best_model.fit(train_x,train_y)
    accuracy = metrics.accuracy_score(test_y,best_model.predict(test_x))
    conn = sqlite3.connect('./databases/accuracy_student.db')
    c = conn.cursor()
    command = 'INSERT INTO student_accuracy VALUES (\'' + request.form['student'] +'\','+str(accuracy)+')'
    c.execute(command)
    conn.commit()
    table = c.execute('select * from student_accuracy order by accuracy desc');
    data = []
    [data.append({'name':name,'accuracy':accuracy})
            for (name,accuracy) in table]
    conn.close()
    return render_template('leaderboard.html',data=data) 

