from flask import Flask,render_template,request
from sklearn import svm,datasets,metrics
from sklearn.model_selection import train_test_split
import numpy as np
import datetime
import matplotlib.pyplot as plt
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
    
    accuracy,x_sets,y_sets = get_accuracy(best_model,X,Y)
    xs,ys,classification = get_classification_boundry_params(best_model,X,Y)
    
    image_name = save_plot(xs,ys,classification,x_sets,y_sets)
    
    data = append_table(accuracy,image_name)

    return render_template('leaderboard.html',data=data) 

def get_accuracy(best_model,X,Y):
    train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.4)
    best_model.fit(train_x,train_y)
    return (metrics.accuracy_score(test_y,best_model.predict(test_x)),[train_x,test_x],[train_y,test_y])

def get_classification_boundry_params(best_model,X,Y):
    X_min,X_max = X[:,0].min(),X[:,0].max()
    padding = (X_max - X_min) / 10
    X_max,X_min = X_max + padding, X_min - padding

    Y_min,Y_max = X[:,1].min(),X[:,1].max() 
    padding = (Y_max - Y_min) / 10
    Y_max, Y_min = Y_max + padding, Y_min - padding
    
    xs,ys = np.meshgrid(np.arange(X_min,X_max,0.02),np.arange(Y_min,Y_max,0.02))
    classification = best_model.predict(np.c_[xs.ravel(),ys.ravel()])
    classification = classification.reshape(xs.shape)
    return (xs,ys,classification)

def save_plot(xs,ys,classification,x_sets,y_sets):
    
    plt.pcolormesh(xs,ys,classification)
    for marker,x,y in zip(['.','o'],x_sets,y_sets):
        for target,label,color in zip(range(3),['I. setosa','I. versicolor','I. virginica'],['y','r','b']) :
            indices = np.argwhere(y == target)
            plt.scatter(x[indices,0],x[indices,1],label=label,c=color,alpha=0.5,marker=marker)

    plt.xlabel(datasets.load_iris().feature_names[int(request.form['feature1'])])
    plt.ylabel(datasets.load_iris().feature_names[int(request.form['feature2'])])
    plt.legend()
    fig = plt.gcf()
    plt.draw()
    image_name = format('{:%Y%m%d%H%M%S}'.format(datetime.datetime.now()))+'.png'
    fig.savefig('static/'+image_name)
    plt.close()
    return image_name

def append_table(accuracy,image_name):
    conn = sqlite3.connect('./databases/accuracy_student.db')
    c = conn.cursor()
    #(name text,accuracy real, image text)
    command = 'INSERT INTO accuracy_student VALUES (\'' + request.form['student'] +'\','+str(accuracy)+',\''+image_name+'\')'
    c.execute(command)
    conn.commit()
    table = c.execute('select * from accuracy_student order by accuracy desc');
    data = []
    [data.append({'name':name,'accuracy':accuracy,'image':image})
            for (name,accuracy,image) in table]
    conn.close()
    return data
