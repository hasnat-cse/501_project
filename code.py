import pandas as pd
import re

from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


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

        else:
            words = line.split('\t')

            if len(words) == 2 and words[0] != 'meta':
                sentence.append(tuple(words))

    return sentences


def penn_to_wn(tag):
    """
    Convert between the PennTreebank tags to simple Wordnet tags
    """
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


def get_sentiment(word, tag):
    """ returns list of pos neg and objective score. But returns empty list if not present in senti wordnet. """

    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []

    lemmatizer = WordNetLemmatizer()

    lemma = lemmatizer.lemmatize(word, pos=wn_tag)
    if not lemma:
        return []

    synsets = wn.synsets(word, pos=wn_tag)
    if not synsets:
        return []

    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())

    return [swn_synset.pos_score(), swn_synset.neg_score(), swn_synset.obj_score()]


def get_eng_words(sentence):
    eng_words = []

    for word_senti_tuple in sentence:
        if word_senti_tuple[1] == "Eng":
            eng_words.append(word_senti_tuple[0])

    return eng_words


def main():
    # processed_texts = preprocess_data()

    sentences = read_word_label_file()

    ps = PorterStemmer()
    for sentence in sentences:
        words = get_eng_words(sentence)
        pos_val = pos_tag(words)
        senti_val = [get_sentiment(x, y) for (x, y) in pos_val]
        print(words)
        print(senti_val)


if __name__ == "__main__":
    main()
