import os
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer

columns = ['person_couple', 'conversation', 'explanation']

df = pd.read_csv('./dataset/explanation_toxic_conversation.csv')
df = df[columns]

label_encoder = LabelEncoder()
df['person_couple'] = label_encoder.fit_transform(df['person_couple'])

X_train, X_test, y_train, y_test = train_test_split(df['conversation'], df['person_couple'], test_size=0.2, stratify=df['person_couple'], random_state=42)

#model = SentenceTransformer('LaBSE')

model_name = 'intfloat/multilingual-e5-large'
model_name_2 = 'intfloat_multilingual-e5-large_normalized'

model = SentenceTransformer(model_name)

X_train_file = './utility/X_train_emb_' + model_name_2 + '.npy'
X_test_file = './utility/X_test_emb_' + model_name_2 + '.npy'

if os.path.exists(X_train_file) and os.path.exists(X_test_file):
	X_train_emb = np.load(X_train_file)
	X_test_emb = np.load(X_test_file)
else:
	X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True, normalize_embeddings=True)
	X_test_emb = model.encode(X_test.tolist(), show_progress_bar=True, normalize_embeddings=True)
	np.save(X_train_file, X_train_emb)
	np.save(X_test_file, X_test_emb)


scaler = MinMaxScaler()

X_train_emb = scaler.fit_transform(X_train_emb)
X_test_emb = scaler.transform(X_test_emb)


print('=== Embedding Model: ' + model_name + ' ===')


param_knn = {
	'n_neighbors': [20, 25, 30, 35],
	'algorithm': ['ball_tree', 'kd_tree', 'brute'],
	'weights': ['uniform', 'distance'],
	'p': [3, 4, 5, 6]
}

knn = GridSearchCV(KNeighborsClassifier(), param_knn, cv=10, scoring='accuracy')
knn.fit(X_train_emb, y_train)
y_pred_knn = knn.predict(X_test_emb)
print('KNN:', round(accuracy_score(y_test, y_pred_knn), 2))


# non normalizzare per lr
'''
param_lr = {
	'C': [10, 11, 12],
	'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
	'max_iter': [200000]
}

lr = GridSearchCV(LogisticRegression(), param_lr, cv=10, scoring='accuracy')
lr.fit(X_train_emb, y_train)
y_pred_lr = lr.predict(X_test_emb)
print('LogisticRegression:', round(accuracy_score(y_test, y_pred_lr), 2))
'''


param_svm = {
	'C': [1.3, 1.4, 1.5],
	'penalty': ['l2'],
	'multi_class': ['ovr', 'crammer_singer'],
	'loss': ['squared_hinge', 'hinge'],
	'max_iter': [10000000]
}

linear_svm = GridSearchCV(LinearSVC(), param_svm, cv=10, scoring='accuracy')
linear_svm.fit(X_train_emb, y_train)
y_pred_svm = linear_svm.predict(X_test_emb)
print('Linear SVM:', round(accuracy_score(y_test, y_pred_svm), 2))



param_svm = {
	'C': [1, 1.5, 2, 5],
	'kernel': ['poly', 'rbf', 'sigmoid'],
	'gamma': ['scale', 'auto'],
	'max_iter': [10000000]
}

svm = GridSearchCV(SVC(), param_svm, cv=10, scoring='accuracy')
svm.fit(X_train_emb, y_train)
y_pred_svm = svm.predict(X_test_emb)
print('SVM:', round(accuracy_score(y_test, y_pred_svm), 2))



param_rf = {
	'n_estimators': [500, 700],
	'max_depth': [20, 25],
	'criterion': ['gini', 'entropy', 'log_loss'],
	'max_features': ['sqrt']
}

rf = GridSearchCV(RandomForestClassifier(random_state=42), param_rf, cv=10, scoring='accuracy')
rf.fit(X_train_emb, y_train)
y_pred_rf = rf.predict(X_test_emb)
print('RandomForest:', round(accuracy_score(y_test, y_pred_rf), 2))



param_mlp = {
	'hidden_layer_sizes': [(512,), (256,)],
	'alpha': [0.0001],
	'learning_rate': ['constant', 'adaptive'],
	'max_iter': [10000],
	'activation': ['tanh', 'logistic'],
	'solver': ['sgd', 'adam'],
	'early_stopping': [True, False]
}

mlp = GridSearchCV(MLPClassifier(random_state=42), param_mlp, cv=10, scoring='accuracy')
mlp.fit(X_train_emb, y_train)
y_pred_mlp = mlp.predict(X_test_emb)
print('MLP:', round(accuracy_score(y_test, y_pred_mlp), 2))


print("Best params KNN:", knn.best_params_)
#print("Best params LR:", lr.best_params_)
print("Best params Linear SVM:", linear_svm.best_params_)
print("Best params SVM:", svm.best_params_)
print("Best params RF:", rf.best_params_)
print("Best params MLP:", mlp.best_params_)