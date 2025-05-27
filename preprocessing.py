# preprocessing.py
import re
import string
import unidecode
import emoji
import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin

class RemoveConsecutiveSpaces(BaseEstimator, TransformerMixin): # Xoá các khoảng trắng liên tiếp
    def transform(self, x): return [' '.join(s.split()) for s in x]
    def fit(self, x, y=None): return self

class RemovePunct(BaseEstimator, TransformerMixin): # Xoá các dấu câu
    def transform(self, x): return [re.sub('[^A-Za-z0-9 ]+', '', s) for s in x]
    def fit(self, x, y=None): return self

class Lowercase(BaseEstimator, TransformerMixin): # Chuyển đổi thành chữ thường
    def transform(self, x): return [s.lower() for s in x]
    def fit(self, x, y=None): return self

class RemoveTone(BaseEstimator, TransformerMixin): # Xoá dấu tiếng Việt
    def transform(self, x): return [unidecode.unidecode(s) for s in x]
    def fit(self, x, y=None): return self

class NumWordsCharsFeature(BaseEstimator, TransformerMixin): # Đếm số từ và số ký tự
    def transform(self, x):
        count_chars = sp.csr_matrix([len(s) for s in x], dtype=np.float64).T
        count_words = sp.csr_matrix([len(s.split()) for s in x], dtype=np.float64).T
        return sp.hstack([count_chars, count_words])
    def fit(self, x, y=None): return self

class ExclamationMarkFeature(BaseEstimator, TransformerMixin): # Đếm số dấu chấm than và dấu hỏi
    def transform(self, x):
        counts = [(s.count('!') + s.count('?')) / (1 + len(s.split())) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).T
    def fit(self, x, y=None): return self

class NumCapitalLettersFeature(BaseEstimator, TransformerMixin): # Đếm số chữ hoa
    def transform(self, x):
        counts = [sum(1 for c in s if c.isupper()) / (1 + len(s)) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).T
    def fit(self, x, y=None): return self

class NumLowercaseLettersFeature(BaseEstimator, TransformerMixin): # Đếm số chữ thường
    def transform(self, x): 
        counts = [sum(1 for c in s if c.islower()) / (1 + len(s)) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).T
    def fit(self, x, y=None): return self

class NumPunctsFeature(BaseEstimator, TransformerMixin): # Đếm số dấu câu
    def transform(self, x):
        counts = [sum(1 for c in s if c in string.punctuation) / (1 + len(s)) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).T
    def fit(self, x, y=None): return self

class NumEmojiFeature(BaseEstimator, TransformerMixin): # Đếm số emoji
    def transform(self, x):
        counts = [len([c for c in s if c in emoji.EMOJI_DATA]) / (1 + len(s.split())) for s in x]
        return sp.csr_matrix(counts, dtype=np.float64).T
    def fit(self, x, y=None): return self

def removeMissingValue(df):
    df = df.dropna()
    return df
def OverSampling(df, target_col):
    from imblearn.over_sampling import RandomOverSampler
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    ros = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = ros.fit_resample(X, y)
    
    return X_resampled, y_resampled