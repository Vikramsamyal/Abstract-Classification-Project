!pip install scikit-learn
!pip install pandas
!pip install numpy

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score


# For loading the datasets with column names
column_names = ['abstract', 'domain']
train_df = pd.read_csv('/content/train.csv', names=column_names)
val_df = pd.read_csv('/content/validation.csv', names=column_names)

# To Initialize label encoder
label_encoder = LabelEncoder()
train_df['label'] = label_encoder.fit_transform(train_df['domain'])
val_df['label'] = label_encoder.transform(val_df['domain'])

# Defining Text preprocessing function to lowercase text
def preprocess_text(text):
    text = text.lower()
    return text


# Apply preprocessing
train_df['abstract'] = train_df['abstract'].apply(preprocess_text)
val_df['abstract'] = val_df['abstract'].apply(preprocess_text)

# Split features and labels
X_train, y_train = train_df['abstract'], train_df['label']
X_val, y_val = val_df['abstract'], val_df['label']


# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=12500)

# Fit and transform the training data, transform the validation data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_val_tfidf = tfidf_vectorizer.transform(X_val)

# Initialize SVM model
svm_model = SVC(kernel='linear', C=1.0)

# Train the model
svm_model.fit(X_train_tfidf, y_train)

# Make predictions on the validation data
y_pred = svm_model.predict(X_val_tfidf)

# Evaluating the model
f1 = f1_score(y_val, y_pred, average='weighted')
print(f"Weighted F1 Score: {f1:.4f}")

# classification report
print(classification_report(y_val, y_pred, target_names=label_encoder.classes_))
