import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sentence_transformers import SentenceTransformer

columns = ['person_couple', 'conversation', 'explanation']

df = pd.read_csv('./dataset/explanation_toxic_conversation.csv')
df = df[columns]

label_encoder = LabelEncoder()
df['person_couple'] = label_encoder.fit_transform(df['person_couple'])

X_train, X_test, y_train, y_test = train_test_split(
	df['conversation'], df['person_couple'],
	test_size=0.2, stratify=df['person_couple'], random_state=42
)

"""
embedding_models = [
	'distiluse-base-multilingual-cased-v1',
	'paraphrase-multilingual-MiniLM-L12-v2',
	'LaBSE',
	'all-MiniLM-L6-v2',
]
"""
embedding_models = [
	'intfloat/multilingual-e5-large',
	'thenlper/gte-large',
	'sentence-transformers/paraphrase-xlm-r-multilingual-v1'
]

param_knn = {
	'n_neighbors': [10, 12, 15, 20],
	'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
	'weights': ['uniform', 'distance'],
	'p': [1, 2, 3]
}

param_lr = {
	'C': [5, 10, 20, 50],
	'solver': ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
	'max_iter': [200000]
}

param_svm = {
	'C': [0.5, 0.8, 1, 1.5, 2],
	'penalty': ['l2'],
	'multi_class': ['ovr'],
	'loss': ['squared_hinge'],
	'max_iter': [1000000]
}

param_rf = {
	'n_estimators': [500, 1000],
	'max_depth': [25],
	'criterion': ['gini', 'entropy', 'log_loss'],
	'max_features': ['sqrt']
}

param_mlp = {
	'hidden_layer_sizes': [(100,), (128, 64)],
	'alpha': [0.0001],
	'learning_rate': ['constant', 'adaptive'],
	'max_iter': [10000],
	'activation': ['tanh', 'logistic'],
	'solver': ['sgd', 'adam'],
	'early_stopping': [True]
}

classifiers = {
	'KNN': (KNeighborsClassifier(), param_knn),
	'LogisticRegression': (LogisticRegression(), param_lr),
	'SVM': (LinearSVC(), param_svm),
	'RandomForest': (RandomForestClassifier(random_state=42), param_rf),
	'MLP': (MLPClassifier(random_state=42), param_mlp)
}

for model_name in embedding_models:
	
	print(f'\n=== Embedding Model: {model_name} ===')
	
	model = SentenceTransformer(model_name)
	
	if 'e5' in model_name or 'gte' in model_name:
		X_train_prompted = ['query: ' + text for text in X_train]
		X_test_prompted = ['query: ' + text for text in X_test]
	else:
		X_train_prompted = X_train.tolist()
		X_test_prompted = X_test.tolist()
	
	X_train_emb = model.encode(X_train.tolist(), show_progress_bar=True)
	X_test_emb = model.encode(X_test.tolist(), show_progress_bar=True)

	for clf_name, (clf, params) in classifiers.items():
		grid = GridSearchCV(clf, params, cv=5, scoring='accuracy')
		grid.fit(X_train_emb, y_train)
		y_pred = grid.predict(X_test_emb)
		acc = accuracy_score(y_test, y_pred)
		print(f'{clf_name}: Accuracy = {round(acc, 2)}')