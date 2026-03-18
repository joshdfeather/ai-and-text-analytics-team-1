import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from datasets import load_dataset

# Stage 1: Load Data 
# --------------------------------------
dataset = load_dataset("pubmed_qa", "pqa_labeled")

# Stage 2: Text representation using simple strings and n-gram vectorizer
# -------------------------------------- 
records = []
for row in dataset['train']:
    text = (
        f"Question: {row['question']} "
        f"Context: {' '.join(row['context']['contexts'])}"
    )
    label = row['final_decision']
    records.append({'text': text, 'label': label})
df = pd.DataFrame(records)

X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42
)

tfidf = CountVectorizer(stop_words='english', ngram_range=(1, 2))
X_train = tfidf.fit_transform(X_train)
X_test = tfidf.transform(X_test)

# Stage 3: Classification using balanced logistic regression 
# --------------------------------------

model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted')}")
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, xticklabels=model.classes_, yticklabels=model.classes_, cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Baseline Model Confusion Matrix')
plt.show()
