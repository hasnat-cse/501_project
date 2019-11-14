from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def pre_process(file_path):
    f = open(file_path, "r", encoding="utf-8")

    contents = f.read()

    data = re.compile("\n").split(contents)
    # print(data)

    return data


def main():
    texts = pd.read_csv("data/training_data.csv", encoding='latin1')

    texts.fillna(0)

    # print(texts.head())
    # print(texts.shape)

    x = texts.iloc[:, 0].values
    y = texts.iloc[:, 1].values

    processed_texts = []
    for sentence in range(0, len(x)):

        # Remove all the special characters
        processed_text = re.sub(r'\W', ' ', str(x[sentence]))

        # remove all single characters
        processed_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_text)

        # Remove single characters from the start
        processed_text = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_text)

        # Substituting multiple spaces with single space
        processed_text = re.sub(r'\s+', ' ', processed_text, flags=re.I)

        # Removing prefixed 'b'
        processed_text = re.sub(r'^b\s+', '', processed_text)

        # Converting to Lowercase
        processed_text = processed_text.lower()

        processed_texts.append(processed_text)

    tfidfconverter = TfidfVectorizer(max_features=2000, min_df=5, max_df=0.7, stop_words=stopwords.words('english'))
    x = tfidfconverter.fit_transform(processed_texts).toarray()

    y = pd.DataFrame(y).fillna("neutral")

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    text_classifier = RandomForestClassifier(n_estimators=100, random_state=0)
    text_classifier.fit(x_train, y_train)

    predictions = text_classifier.predict(x_test)

    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    print(accuracy_score(y_test, predictions))


if __name__ == "__main__":
    main()
