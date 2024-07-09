import os
import re
import numpy as np
import pandas as pd
import spacy
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
import emoji
from textblob import TextBlob
from gensim.models import KeyedVectors
from scipy.sparse import hstack, csr_matrix
import wandb
import joblib
import matplotlib.pyplot as plt

# Set W&B API key
os.environ["WANDB_API_KEY"] = "711edcd96cef1e4ed489e2e9e21732f93abb35ae"

# Initialize Weights & Biases
wandb.init(project="text_classification")

# Load spaCy model for advanced NLP tasks
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.max_length = 2000000

# Load Word2Vec embeddings
word2vec_path = "GoogleNews-vectors-negative300.bin"
word2vec_embeddings = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

# Load the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=3000, stop_words='english')

def clean_text(text):
    text = emoji.demojize(text, delimiters=("", " "))
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text(text, nlp_model):
    doc = nlp_model(clean_text(text))
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def replace_slang(text, slang_dict):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in slang_dict.keys()) + r')\b')
    return pattern.sub(lambda x: slang_dict[x.group()], text)

def document_vector(word_list, embeddings):
    valid_words = [word for word in word_list if word in embeddings.key_to_index]
    if not valid_words:
        return np.zeros(embeddings.vector_size)
    return np.mean([embeddings[word] for word in valid_words], axis=0)

def extract_features(df, tfidf_vectorizer, word2vec_embeddings):
    tfidf_features = tfidf_vectorizer.fit_transform(df['processed_text'])
    sentiment = np.array(df['sentiment']).reshape(-1, 1)
    word2vec_features = np.stack(df['processed_text'].apply(lambda doc: document_vector(doc.split(), word2vec_embeddings)).values)
    return hstack([tfidf_features, csr_matrix(sentiment), csr_matrix(word2vec_features)])

def load_and_shuffle_data(filepath):
    df = pd.read_csv(filepath)
    df.dropna(subset=['text'], inplace=True)
    df = df.sample(frac=1).reset_index(drop=True)  # Shuffle the dataset
    df['processed_text'] = df['text'].apply(lambda x: preprocess_text(x, nlp))
    df['sentiment'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)

    # Fit the TF-IDF vectorizer and save it
    tfidf_vectorizer.fit(df['processed_text'])
    joblib.dump(tfidf_vectorizer, 'tfidf_vectorizerBERT.pkl')

    return df

class BertWithAdditionalFeatures(torch.nn.Module):
    def __init__(self, bert_model, additional_feature_size):
        super(BertWithAdditionalFeatures, self).__init__()
        self.bert = bert_model
        self.additional_feature_size = additional_feature_size
        self.hidden_size = bert_model.config.hidden_size

        # Adding a placeholder for the linear layer which will be set up later
        self.fc = None

    def forward(self, input_ids, attention_mask, additional_features):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.pooler_output  # Use pooler_output instead of outputs[1]

        # Combine BERT output with additional features
        combined_output = torch.cat((cls_output, additional_features), dim=1)
        print(f"Combined output shape: {combined_output.shape}")

        # If the linear layer is not yet initialized, initialize it now
        if self.fc is None:
            combined_feature_size = combined_output.shape[1]
            self.fc = torch.nn.Linear(combined_feature_size, 2)
            self.fc.to(combined_output.device)  # Move the layer to the same device as the data

        logits = self.fc(combined_output)
        return logits


def prepare_data(df, tfidf_vectorizer, word2vec_embeddings, indices):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    subset_df = df.iloc[indices]
    encoded_batch = tokenizer.batch_encode_plus(subset_df['processed_text'].tolist(), add_special_tokens=True,
                                                max_length=128, padding='max_length', truncation=True,
                                                return_tensors='pt')
    inputs = encoded_batch['input_ids']
    masks = encoded_batch['attention_mask']
    labels = torch.tensor(subset_df['label'].values)
    additional_features = extract_features(subset_df, tfidf_vectorizer, word2vec_embeddings).toarray()
    additional_features = torch.tensor(additional_features, dtype=torch.float)

    # Print the dimensions of the features
    print(f"Inputs shape: {inputs.shape}")
    print(f"Masks shape: {masks.shape}")
    print(f"Additional features shape: {additional_features.shape}")

    return TensorDataset(inputs, masks, additional_features, labels), additional_features.shape[1]


def train_model(train_dataset, val_dataset):
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    additional_feature_size = train_dataset[0][2].shape[0]
    model = BertWithAdditionalFeatures(bert_model, additional_feature_size=additional_feature_size)  # TF-IDF (3000) + sentiment (1) + Word2Vec (300)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    val_dataloader = DataLoader(val_dataset, batch_size=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Calculate class weights
    labels = [label for _, _, _, label in train_dataset]
    class_sample_count = np.array([len(np.where(labels == t)[0]) for t in np.unique(labels)])
    weight = 1. / class_sample_count
    samples_weight = np.array([weight[t] for t in labels])
    class_weights = torch.tensor(weight, dtype=torch.float).to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    total_steps = len(train_dataloader) * 3
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    best_accuracy = 0

    for epoch in range(3):
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_additional_features, b_labels = batch
            model.zero_grad()
            logits = model(b_input_ids, b_input_mask, b_additional_features)
            loss = torch.nn.CrossEntropyLoss(weight=class_weights)(logits, b_labels)
            loss.backward()
            total_train_loss += loss.item()
            optimizer.step()
            scheduler.step()

        avg_train_loss = total_train_loss / len(train_dataloader)
        wandb.log({"train_loss": avg_train_loss})

        model.eval()
        total_eval_accuracy = 0
        total_eval_precision = 0
        total_eval_recall = 0
        total_eval_f1 = 0
        eval_steps = 0

        for batch in val_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_additional_features, b_labels = batch
            with torch.no_grad():
                logits = model(b_input_ids, b_input_mask, b_additional_features)
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            tmp_eval_accuracy = accuracy_score(label_ids, np.argmax(logits, axis=1))
            precision, recall, f1, _ = precision_recall_fscore_support(label_ids, np.argmax(logits, axis=1), average='binary')
            total_eval_accuracy += tmp_eval_accuracy
            total_eval_precision += precision
            total_eval_recall += recall
            total_eval_f1 += f1
            eval_steps += 1

        avg_val_accuracy = total_eval_accuracy / eval_steps
        avg_val_precision = total_eval_precision / eval_steps
        avg_val_recall = total_eval_recall / eval_steps
        avg_val_f1 = total_eval_f1 / eval_steps
        wandb.log({"val_accuracy": avg_val_accuracy, "val_precision": avg_val_precision, "val_recall": avg_val_recall, "val_f1": avg_val_f1})

        if avg_val_accuracy > best_accuracy:
            best_accuracy = avg_val_accuracy
            torch.save(model.state_dict(), './best_model.pt')  # Save the best model state

    return best_accuracy

def cross_validate_model(df, k=5):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold = 0
    best_accuracies = []

    for train_index, val_index in skf.split(df, df['label']):
        print(f"Fold {fold + 1}")
        train_dataset = prepare_data(df, tfidf_vectorizer, word2vec_embeddings, train_index)
        val_dataset = prepare_data(df, tfidf_vectorizer, word2vec_embeddings, val_index)
        # Train the model on the current fold
        best_accuracy = train_model(train_dataset, val_dataset)
        best_accuracies.append(best_accuracy)
        fold += 1

    avg_best_accuracy = np.mean(best_accuracies)
    print(f"Average Best Accuracy across folds: {avg_best_accuracy}")
    wandb.log({"avg_best_accuracy": avg_best_accuracy})

def evaluate_model(test_dataset):
    bert_model = BertModel.from_pretrained('bert-base-uncased')
    model = BertWithAdditionalFeatures(bert_model, additional_feature_size=3301)
    model.load_state_dict(torch.load('./best_model.pt'))
    model.eval()

    test_dataloader = DataLoader(test_dataset, batch_size=16)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    total_test_accuracy = 0
    total_test_precision = 0
    total_test_recall = 0
    total_test_f1 = 0
    test_steps = 0

    all_labels = []
    all_predictions = []

    for batch in test_dataloader:
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_additional_features, b_labels = batch
        with torch.no_grad():
            logits = model(b_input_ids, b_input_mask, b_additional_features)
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        all_labels.extend(label_ids)
        all_predictions.extend(np.argmax(logits, axis=1))
        tmp_test_accuracy = accuracy_score(label_ids, np.argmax(logits, axis=1))
        precision, recall, f1, _ = precision_recall_fscore_support(label_ids, np.argmax(logits, axis=1), average='binary')
        total_test_accuracy += tmp_test_accuracy
        total_test_precision += precision
        total_test_recall += recall
        total_test_f1 += f1
        test_steps += 1

    avg_test_accuracy = total_test_accuracy / test_steps
    avg_test_precision = total_test_precision / test_steps
    avg_test_recall = total_test_recall / test_steps
    avg_test_f1 = total_test_f1 / test_steps
    wandb.log({"test_accuracy": avg_test_accuracy, "test_precision": avg_test_precision, "test_recall": avg_test_recall, "test_f1": avg_test_f1})

    print(f"Test Accuracy: {avg_test_accuracy}")
    print(f"Test Precision: {avg_test_precision}")
    print(f"Test Recall: {avg_test_recall}")
    print(f"Test F1: {avg_test_f1}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['unrelated', 'pro-ed'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("ConfusionMatrix.png")
    plt.show()

# The rest of the script should remain the same, making sure the prepare_data function and others are aligned with the updated class.

if __name__ == "__main__":
    filepath = 'dateProEdsiNotProED.csv'
    df = load_and_shuffle_data(filepath)

    # Load the TF-IDF vectorizer
    tfidf_vectorizer = joblib.load('tfidf_vectorizerBERT.pkl')

    cross_validate_model(df, k=5)

    # Prepare data for final evaluation
    train_dataset, val_dataset, test_dataset = prepare_data(df, tfidf_vectorizer, word2vec_embeddings, np.arange(len(df)))
    train_model(train_dataset, val_dataset)
    evaluate_model(test_dataset)