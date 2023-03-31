#!/usr/bin/env python
# coding: utf-8

# ## Aquí importamos la librerias que vamos a utilizar

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split


# In[2]:


columns = ['Sepal length', 'Sepal width', 'Petal length', 'Petal width', 'Class_labels'] 
# Load the data
df = pd.read_csv('iris.data', names=columns)
df.head()


# In[3]:


df.describe()


# ## Graficamos aquí los datos y encontramos las relaciones

# In[4]:


sns.pairplot(df, hue='Class_labels')


# ## Separamos la data en train y test

# In[5]:


data = df.values
X = data[:,(2,3)]
Y = data[:,4]


# In[6]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


# In[7]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

iris_pipeline = Pipeline([("std_scaler", StandardScaler())])


# In[8]:


X_train_tr = iris_pipeline.fit_transform(X_train)


# ## Guardamos el .sav del pipeline

# In[9]:


from joblib import dump
dump(iris_pipeline, "iris_pipeline.sav")


# ## Implementamos nuestro primer modelo "Support Machine Vector"

# In[10]:


from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000], 
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf']} 

SVC = SVC()

SVC_gridCV = GridSearchCV(SVC, param_grid, refit = True, verbose = 3, cv=5)
  
SVC_gridCV.fit(X_train_tr, y_train)


# In[11]:


SVC_model=SVC_gridCV.best_estimator_
X_test_tr = iris_pipeline.fit_transform(X_test)
final_prediction_svc = SVC_model.predict(X_test_tr)


# In[12]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
conf_matrix = confusion_matrix(y_test, final_prediction_svc )

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)


# In[13]:


print('Accuracy: %.3f' % accuracy_score(y_test, final_prediction_svc))
print('Precision: %.3f' % precision_score(y_test, final_prediction_svc, average='weighted'))
print('Recall: %.3f' % recall_score(y_test, final_prediction_svc, average='weighted'))
print('F1 Score: %.3f' % f1_score(y_test, final_prediction_svc, average='weighted'))


# In[14]:


from sklearn.metrics import classification_report
print(classification_report(y_test, final_prediction_svc))


# In[15]:


dump(SVC_model, "SVM.sav")


# ## Implementamos nuestro primer modelo "Logistic Regression"

# In[16]:


from sklearn.linear_model import LogisticRegression

param_grid = {"solver": ["liblinear"], "penalty": ["l1", "l2"], "C": np.logspace(-3,3,7)}

log_reg = LogisticRegression()

log_reg_gridCV = GridSearchCV(log_reg,param_grid, cv = 10)
log_reg_gridCV.fit(X_train_tr,y_train)


# In[17]:


Log_reg_model=log_reg_gridCV.best_estimator_
X_test_tr = iris_pipeline.fit_transform(X_test)
final_prediction_log = Log_reg_model.predict(X_test_tr)


# In[18]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
conf_matrix = confusion_matrix(y_test, final_prediction_log)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)


# In[19]:


print('Accuracy: %.3f' % accuracy_score(y_test, final_prediction_log))
print('Precision: %.3f' % precision_score(y_test, final_prediction_log, average='weighted'))
print('Recall: %.3f' % recall_score(y_test, final_prediction_log, average='weighted'))
print('F1 Score: %.3f' % f1_score(y_test, final_prediction_log, average='weighted'))


# In[20]:


from sklearn.metrics import classification_report
print(classification_report(y_test, final_prediction_log))


# In[23]:


dump(Log_reg_model, "Log_reg.sav")


# ## Implementamos nuestro primer modelo "Decision Tree"

# In[21]:


from sklearn.tree import DecisionTreeClassifier
param_grid = {'max_features': ['sqrt'],
              'ccp_alpha': [0.1, .01, .001],
              'max_depth' : [5, 6, 7, 8, 9],
              'criterion' :['gini', 'entropy']
             }
dec_tree = DecisionTreeClassifier(random_state=42)
dec_gridCV = GridSearchCV(dec_tree, param_grid, cv=10)
dec_gridCV.fit(X_train_tr,y_train)


# In[22]:


dec_tree_model=dec_gridCV.best_estimator_
X_test_tr = iris_pipeline.fit_transform(X_test)
final_prediction_tree = dec_tree_model.predict(X_test_tr)


# In[23]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
conf_matrix = confusion_matrix(y_test, final_prediction_tree)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)


# In[24]:


print('Accuracy: %.3f' % accuracy_score(y_test, final_prediction_tree))
print('Precision: %.3f' % precision_score(y_test, final_prediction_tree, average='weighted'))
print('Recall: %.3f' % recall_score(y_test, final_prediction_tree, average='weighted'))
print('F1 Score: %.3f' % f1_score(y_test, final_prediction_tree, average='weighted'))


# In[25]:


from sklearn.metrics import classification_report
print(classification_report(y_test, final_prediction_tree))


# In[26]:


dump(dec_tree_model, "Dec_tree.sav")


# ## Implementamos nuestro primer modelo "Voting Classifier"

# In[27]:


from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

estimators = [
['Logistic Regression: ', LogisticRegression(random_state=42)],
['Support Vector Machine :', SVC(gamma ='auto', probability = True, random_state=42)],
['Random Forest :', RandomForestClassifier(random_state=42)]]

voting_clf = VotingClassifier(estimators = estimators, voting = 'hard')
voting_clf.fit(X_train_tr,y_train)


# In[28]:


X_test_tr = iris_pipeline.fit_transform(X_test)
final_prediction_voting_clf = voting_clf.predict(X_test_tr)


# In[29]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
conf_matrix = confusion_matrix(y_test, final_prediction_voting_clf)

fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)


# In[30]:


print('Accuracy: %.3f' % accuracy_score(y_test, final_prediction_voting_clf))
print('Precision: %.3f' % precision_score(y_test, final_prediction_voting_clf, average='weighted'))
print('Recall: %.3f' % recall_score(y_test, final_prediction_voting_clf, average='weighted'))
print('F1 Score: %.3f' % f1_score(y_test, final_prediction_voting_clf, average='weighted'))


# In[31]:


from sklearn.metrics import classification_report
print(classification_report(y_test, final_prediction_voting_clf))


# In[32]:


dump(voting_clf, "vot_clf.sav")

