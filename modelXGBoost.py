import os
import re
import numpy as np
import pandas as pd
import sns as sns
import spacy
from matplotlib import pyplot as plt
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score, learning_curve
from sklearn.metrics import classification_report
import xgboost as xgb
from gensim.models import KeyedVectors
from scipy.sparse import hstack, csr_matrix
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib
import emoji

# Load spaCy English model with only the components needed for lemmatization
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.max_length = 2000000

def load_word2vec_embeddings(path):
    try:
        return KeyedVectors.load_word2vec_format(path, binary=True)
    except Exception as e:
        print(f"Error loading Word2Vec model: {e}")
        return None

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        df.dropna(subset=['text'], inplace=True)
        return df
    except Exception as e:
        print(f"Error reading data from {filepath}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on failure

def clean_text(text):
    """Clean and preprocess text by removing URLs, non-alphabetic characters, and extra spaces."""
    text = emoji.demojize(text, delimiters=("", " "))
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def preprocess_text(text, slang_dict):
    """Process text by replacing slang and removing stopwords and punctuation."""
    if text is None or text.strip() == "":
        return None
    text = replace_slang(text, slang_dict)
    text = clean_text(text)
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_stop and token.is_alpha)

def replace_slang(text, slang_dict):
    """Replace slang terms in text using the provided slang dictionary."""
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in slang_dict.keys()) + r')\b')
    return pattern.sub(lambda x: slang_dict[x.group()], text)

def load_keywords_from_files(directory):
    """Load keywords from text files within the specified directory."""
    keywords_dict = {}
    try:
        for filename in os.listdir(directory):
            if filename.endswith('.txt'):
                category = filename[:-4]
                path = os.path.join(directory, filename)
                with open(path, 'r') as file:
                    keywords = [line.strip() for line in file if line.strip()]
                keywords_dict[category] = keywords
    except Exception as e:
        print(f"Failed to load keywords due to: {e}")
    return keywords_dict

def categorize_post(text, keywords):
    """Categorize text based on the presence of keywords."""
    scores = {category: sum(keyword in text for keyword in keywords[category]) for category in keywords}
    return max(scores, key=scores.get, default='unknown')

def extract_features(df, tfidf_vectorizer, word2vec_embeddings):
    """Extract TF-IDF and Word2Vec features from text data."""
    tfidf_features = tfidf_vectorizer.fit_transform(df['processed_text'])
    sentiment = np.array(df['sentiment']).reshape(-1, 1)
    word2vec_features = np.stack(df['processed_text'].apply(lambda doc: document_vector(doc.split(), word2vec_embeddings)).values)
    return hstack([tfidf_features, csr_matrix(sentiment), csr_matrix(word2vec_features)])

def document_vector(word_list, embeddings):
    """Generate document vector by averaging word vectors in the document."""
    # Filter valid words based on the new Gensim KeyedVectors API
    valid_words = [word for word in word_list if word in embeddings.key_to_index]
    if not valid_words:
        return np.zeros(embeddings.vector_size)
    return np.mean([embeddings[word] for word in valid_words], axis=0)


def evaluate_model(model, X, y):
    """Evaluate model using cross-validation and print the results."""
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validated scores: {scores}")
    print(f"Average score: {np.mean(scores)}")


def plot_learning_curve(estimator, X, y, cv, scoring='accuracy'):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.figure()
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_feature_importances(model, tfidf_vectorizer):
    feature_importances = model.feature_importances_
    sorted_idx = np.argsort(feature_importances)[::-1][:20]  # Top 20 features
    features = np.array(tfidf_vectorizer.get_feature_names_out())[sorted_idx]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances[sorted_idx], y=features)
    plt.title('Top 20 Feature Importances')
    plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(y_test, y_pred, classes):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.show()

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def plot_roc_curve(y_test, y_score, classes):
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    n_classes = y_test_bin.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], roc_auc[i]))
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

from sklearn.metrics import precision_recall_curve, average_precision_score

def plot_precision_recall_curve(y_test, y_score, classes):
    y_test_bin = label_binarize(y_test, classes=range(len(classes)))
    n_classes = y_test_bin.shape[1]

    precision = dict()
    recall = dict()
    average_precision = dict()
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        average_precision[i] = average_precision_score(y_test_bin[:, i], y_score[:, i])

    plt.figure()
    colors = ['navy', 'turquoise', 'darkorange']
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2,
                 label='Precision-Recall curve of class {0} (area = {1:0.2f})'
                       ''.format(classes[i], average_precision[i]))

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

def plot_class_distribution(y, classes):
    plt.figure(figsize=(8, 6))
    sns.countplot(x=y, palette="viridis")
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.xticks(ticks=range(len(classes)), labels=classes, rotation=45)
    plt.show()
import shap

def plot_shap_values(model, X):
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X)

def main():
    filepath = 'archive (4)/proED_full_dataset.csv'
    word2vec_path = "GoogleNews-vectors-negative300.bin"
    keywords_directory = 'keywords'
    slang_dict = {'lol': 'laughing out loud', 'brb': 'be right back', 'gtg': 'got to go', 'imo': 'in my opinion'}

    word2vec_embeddings = load_word2vec_embeddings(word2vec_path)
    keywords = load_keywords_from_files(keywords_directory)
    df = load_data(filepath)

    df['processed_text'] = df['Text'].apply(lambda x: preprocess_text(x, slang_dict))
    df['sentiment'] = df['Text'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['category'] = df['processed_text'].apply(lambda x: categorize_post(x, keywords))

    df.to_csv('labeled_dataset.csv', index=False)

    tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')
    tfidf_vectorizer.fit_transform(df['processed_text'])
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
    X = extract_features(df, tfidf_vectorizer, word2vec_embeddings)
    y = pd.factorize(df['category'])[0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    smote = SMOTE(random_state=42)
    model = xgb.XGBClassifier(objective='multi:softmax', use_label_encoder=False)

    pipeline = Pipeline([
        ('smote', smote),
        ('classifier', model)
    ])
    parameters = {
        'classifier__max_depth': [3, 5],
        'classifier__learning_rate': [0.1, 0.2],
        'classifier__n_estimators': [100, 200]
    }
    grid_search = RandomizedSearchCV(pipeline, parameters, cv=3, scoring='accuracy', n_iter=4)
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.best_estimator_.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=np.unique(df['category'])))
    print("Best parameters found: ", grid_search.best_params_)

    # Predictions and classification report
    y_pred = grid_search.best_estimator_.predict(X_test)
    y_score = grid_search.best_estimator_.predict_proba(X_test)
    print(classification_report(y_test, y_pred, target_names=np.unique(df['category'])))

    # Plot feature importances
    plot_feature_importances(grid_search.best_estimator_.named_steps['category'], tfidf_vectorizer)

    # Plot learning curve
    plot_learning_curve(grid_search.best_estimator_, X_train, y_train, cv=3)

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, np.unique(df['category']))

    # Plot ROC curve
    plot_roc_curve(y_test, y_score, np.unique(df['category']))

    # Plot precision-recall curve
    plot_precision_recall_curve(y_test, y_score,np.unique(df['category']))

    # Plot class distribution
    plot_class_distribution(y,np.unique(df['category']))

    # Plot SHAP values
    plot_shap_values(grid_search.best_estimator_.named_steps['classifier'], X)

    plot_feature_importances(grid_search.best_estimator_.named_steps['classifier'], tfidf_vectorizer)

    # Plot learning curve
    plot_learning_curve(grid_search.best_estimator_, X_train, y_train, cv=3)

    # Optionally save the model
    joblib.dump(grid_search.best_estimator_, 'final_model.pkl')
    # joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')

if __name__ == "__main__":
    main()

#    neutral       0.90      0.91      0.91      7068
#       pro-ED       0.90      0.88      0.89      6491
# pro-recovery       0.73      0.84      0.78       627
#
#     accuracy                           0.89     14186
#    macro avg       0.85      0.88      0.86     14186
# weighted avg       0.89      0.89      0.89     14186
#
# Best parameters found:  {'classifier__n_estimators': 200, 'classifier__max_depth': 5, 'classifier__learning_rate': 0.1}
#               precision    recall  f1-score   support
#
#      neutral       0.90      0.91      0.91      7068
#       pro-ED       0.90      0.88      0.89      6491
# pro-recovery       0.73      0.84      0.78       627
#
#     accuracy                           0.89     14186
#    macro avg       0.85      0.88      0.86     14186
# weighted avg       0.89      0.89      0.89     14186