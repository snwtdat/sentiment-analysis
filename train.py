# train_model.py
import time
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

from preprocessing import *


# Load data
df_train = pd.read_csv("/Users/softann/sentiment-analysis/dataset/train.csv")
df_test = pd.read_csv("/Users/softann/sentiment-analysis/dataset/test.csv")

removeMissingValue(df_train)

OverSampling(df_train, "label")
X = df_train['comment'].values
y = df_train['label'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

feature_union = FeatureUnion([
    ('custom_features_pipeline', Pipeline([
        ('custom_features', FeatureUnion([
            ('f01', NumWordsCharsFeature()),
            ('f02', NumCapitalLettersFeature()),
            ('f03', ExclamationMarkFeature()),
            ('f04', NumPunctsFeature()),
            ('f05', NumLowercaseLettersFeature()),
            ('f06', NumEmojiFeature())
        ], n_jobs=-1)),
        ('scaler', StandardScaler(with_mean=False))
    ])),
    ('word_char_features_pipeline', Pipeline([
        ('lowercase', Lowercase()),
        ('word_char_features', FeatureUnion([
            ('with_tone', Pipeline([
                ('remove_punct', RemovePunct()),
                ('tf_idf_word', TfidfVectorizer(ngram_range=(1, 4), norm='l2', min_df=2))
            ])),
            ('with_tone_char', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char')),
            ('with_tone_char_wb', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char_wb')),
            ('without_tone', Pipeline([
                ('remove_tone', RemoveTone()),
                ('without_tone_features', FeatureUnion([
                    ('tf_idf', Pipeline([
                        ('remove_punct', RemovePunct()),
                        ('word', TfidfVectorizer(ngram_range=(1, 4), norm='l2', min_df=2))
                    ])),
                    ('tf_idf_char', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char')),
                    ('tf_idf_char_wb', TfidfVectorizer(ngram_range=(1, 6), norm='l2', min_df=2, analyzer='char_wb'))
                ], n_jobs=-1))
            ]))
        ], n_jobs=-1))
    ]))
], n_jobs=-1)

def train_and_evaluate(model, model_name):
    print(f"\nTraining {model_name}...")
    start_time = time.time()

    pipeline = Pipeline([
        ('remove_spaces', RemoveConsecutiveSpaces()),
        ('features', feature_union),
        ('clf', model)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    print(classification_report(y_val, y_pred))
    print(f"\nTime taken: {time.time() - start_time:.2f} seconds")
    cm = confusion_matrix(y_val, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)

    return pipeline

svm_model = train_and_evaluate(SVC(kernel='linear', C=0.2175, class_weight='balanced'), "SVM")
rf_model = train_and_evaluate(RandomForestClassifier(n_estimators=100, class_weight='balanced'), "Random Forest")
lr_model = train_and_evaluate(LogisticRegression(max_iter=1000, class_weight='balanced'), "Logistic Regression")
knn_model = train_and_evaluate(DecisionTreeClassifier(random_state=42, class_weight='balanced'), "Decision Tree")

joblib.dump(svm_model, "svc_model.pkl")
