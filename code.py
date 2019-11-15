import pandas as pd
import re


def preprocess_data():
    data = pd.read_csv('data/training_data.csv', encoding="latin1")

    # print(data.head(10))

    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values

    processed_texts = []
    for sentence in range(0, len(x)):
        processed_text = str(x[sentence])

        # Replace @name by USER
        processed_text = re.sub('@[\w\_]+', "USER", processed_text)

        # Replace #name by HASHTAG
        processed_text = re.sub('#[\w\_]+', "HASHTAG", processed_text)

        # Replace https by URL
        processed_text = re.sub('https[\w\./]*', "URL", processed_text)

        # Replace all the special characters by ' '
        processed_text = re.sub(r'\W', ' ', processed_text)

        # Substituting multiple spaces with single space
        processed_text = re.sub(r'\s+', ' ', processed_text, flags=re.I)

        # Remove single space from the start
        processed_text = re.sub('^[\s]+', '', processed_text)

        # Remove all single characters
        processed_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_text)

        # Remove single characters from the start
        processed_text = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_text)

        # Converting to Lowercase
        processed_text = processed_text.lower()

        # Remove more than 2 repetition of letters in word
        pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
        processed_text = pattern.sub(r"\1", processed_text)

        processed_texts.append(processed_text)

        print(processed_text)

        print("\n")

    return processed_texts


def read_word_label_file():
    f = open('data/train_conll.txt', encoding="latin1")
    data = f.read()

    sentences = []
    sentence = []

    lines = data.split('\n')

    for line in lines:
        if line == '':
            sentences.append(sentence)
            sentence = []

        words = line.split('\t')

        if words[0] != 'meta':
            sentence.append(tuple(words))

    print(len(sentences))

    return sentences


def main():
    # preprocess_data()
    read_word_label_file()


if __name__ == "__main__":
    main()
