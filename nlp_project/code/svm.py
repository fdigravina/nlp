from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sentence_transformers import SentenceTransformer
from sklearn.svm import SVC

columns = ['person_couple', 'conversation', 'explanation']

df = pd.read_csv('./dataset/explanation_toxic_conversation.csv')
df = df[columns]


label_encoder = LabelEncoder()
df['person_couple'] = label_encoder.fit_transform(df['person_couple'])

X_train, X_test, y_train, y_test = train_test_split(df['conversation'], df['person_couple'], test_size=0.2, stratify=df['person_couple'], random_state=42)

model_name = 'intfloat/multilingual-e5-large'
model_name_2 = 'intfloat_multilingual-e5-large_normalized'

model = SentenceTransformer(model_name)

X_train_file = './utility/X_train_emb_' + model_name_2 + '.npy'
X_test_file = './utility/X_test_emb_' + model_name_2 + '.npy'

if os.path.exists(X_train_file) and os.path.exists(X_test_file):
	X_train_emb = np.load(X_train_file)
	X_test_emb = np.load(X_test_file)
else:
	X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True)
	X_test_emb = model.encode(X_test.tolist(), show_progress_bar=True)
	np.save(X_train_file, X_train_emb)
	np.save(X_test_file, X_test_emb)


scaler = MinMaxScaler()

X_train_emb = scaler.fit_transform(X_train_emb)
X_test_emb = scaler.transform(X_test_emb)


param_svm = {
	'C': [15, 20, 25],
	'kernel': ['rbf', 'sigmoid'],
	'gamma': ['auto'],
	'max_iter': [-1],
	'decision_function_shape': ['ovr'],
	'tol': [1e-4],
	'verbose': [False],
	'break_ties': [True]
}

svm = GridSearchCV(SVC(), param_svm, cv=10, scoring='accuracy')
svm.fit(X_train_emb, y_train)

estimator = svm.best_estimator_
y_pred_svm = estimator.predict(X_test_emb)

print('SVM:', round(accuracy_score(y_test, y_pred_svm), 2))
print("Best params SVM:", svm.best_params_)


cm = confusion_matrix(y_test, y_pred_svm)

classes = [
	'Controllore',
	'Dominante',
	'Geloso',
	'Manipolatore',
	'Narcisista',
	'Perfezionista',
	'Violento',
	'Psicopatico',
	'Sadico',
	'Vittimista',
]

sns.heatmap(cm, annot=True, fmt='d', cmap='coolwarm', xticklabels=classes, yticklabels=classes)

plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix',fontsize=16)
plt.tight_layout()
plt.savefig('./images/confusion_matrix_svm.png', bbox_inches='tight')

print('')

print('Micro precision average:', round(precision_score(y_test, y_pred_svm, average='micro'), 2))
print('Macro precision average:', round(precision_score(y_test, y_pred_svm, average='macro'), 2), '\n')

print('Micro recall average:', round(recall_score(y_test, y_pred_svm, average='micro'), 2))
print('Macro recall average:', round(recall_score(y_test, y_pred_svm, average='macro'), 2), '\n')

print('Micro f1 average:', round(f1_score(y_test, y_pred_svm, average='micro'), 2))
print('Macro f1 average:', round(f1_score(y_test, y_pred_svm, average='macro'), 2), '\n')

print('Mean accuracy and std dev:', end=' ')
print(round(svm.cv_results_['mean_test_score'][svm.best_index_], 2), '±', round(svm.cv_results_['std_test_score'][svm.best_index_], 2))