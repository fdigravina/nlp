import os
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sentence_transformers import SentenceTransformer
from sentence_transformers import SentenceTransformer

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

param_rf = {
	'n_estimators': [700],
	'max_depth': [20, 25],
	'criterion': ['gini', 'entropy'],
	'max_features': ['sqrt'],
	'min_samples_split': [2, 5],
	'min_samples_leaf': [1, 3],
	'verbose': [0]
}

rf = GridSearchCV(RandomForestClassifier(random_state=42), param_rf, cv=10, scoring='accuracy')
rf.fit(X_train_emb, y_train)
y_pred_rf = rf.predict(X_test_emb)

print('RandomForest:', round(accuracy_score(y_test, y_pred_rf), 2))
print("Best params MLP:", rf.best_params_)

cm = confusion_matrix(y_test, y_pred_rf)

classes = [
	'Controllore e Isolata',
	'Dominante e Schiavo emotivo',
	'Geloso-Ossessivo e Sottomessa',
	'Manipolatore e Dipendente emotiva',
	'Narcisista e Succube',
	'Perfezionista Critico e Insicura Cronica',
	'Persona violenta e Succube',
	'Psicopatico e Adulatrice',
	'Sadico-Crudele e Masochista',
	'Vittimista e Croccerossina',
]

sns.heatmap(cm, annot=True,fmt='d', cmap='YlGnBu', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.title('Confusion Matrix',fontsize=16)
plt.savefig('./images/confusion_matrix_rf.png')

print('Micro average:', precision_score(y_test, y_pred_rf, average='micro'))
print('Macro average:', precision_score(y_test, y_pred_rf, average='macro'))