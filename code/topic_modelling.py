import os
import pickle  # librairie pour save des modèle de machine learning
import re
from typing import List, TextIO

import numpy as np
import pandas as pd
from french_lefff_lemmatizer.french_lefff_lemmatizer import FrenchLefffLemmatizer as FLF
from nltk import word_tokenize
from nltk.corpus import stopwords
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

lemmatizer = FLF()

path_data = "insert your local path"
path_to_model = os.path.join(path_data, "model_ML")

stop_words = set(list(open(os.path.join(path_data, "french_stopwords.txt"), "r", encoding="utf-8").read().split("\n"))
                 + stopwords.words("french"))


def table_ocr_1880():
    print("récupération des fichiers")
    fichiers = os.listdir(os.path.join(path_data, "blocs"))
    data = pd.DataFrame(columns=["date", "text"])
    for name in fichiers:
        print(name)
        if name.split("-")[0][:2] == '18':
            file: TextIO = open(os.path.join(path_data, "blocs", name), encoding="utf-8")
            text: str = file.read()
            file.close()
            data = data.append({'date': name.split('.')[0], "text": text}, ignore_index=True)
    return data


def build_corpus(df: pd.DataFrame):
    print("nettoyage du texte")
    data: List = []
    for i in range(df.shape[0]):
        print(f"{i}/{df.shape[0]}")
        text = " ".join(re.findall("[A-Za-zâêûîôäëüïöùàçéèÉ\-\.]+", df.text[i]))
        text = re.sub("([a-z])- ", r"\1", text)
        text = re.sub("\-", " ", text)
        text = re.sub("[M]+\. ([A-Z]+[a-zâêûîôäëüïöùàçéè]+(?:\s[A-Z]+[a-zâêûîôäëüïöùàçéè]+)?)", " ", text)
        text = re.sub("\.", " ", text)
        bag_of_words: List[str] = word_tokenize(text.lower(), language="french")
        bag_of_words = [w for w in bag_of_words if 1 <= len(w) < 22]
        bag_of_words = [lemmatizer.lemmatize(w).lower() for w in bag_of_words if w not in stop_words]
        data.append(bag_of_words)
    df.loc[:, "bag_of_words"] = data
    return df


def count_vectorizer(df: pd.DataFrame, p: int):
    print("compte des coccurences")
    data = [" ".join(w) for w in df.bag_of_words]
    vectorizer = CountVectorizer(max_features=p)
    X = vectorizer.fit_transform(data)
    word_frequency_matrix = pd.DataFrame(data=X.toarray(), index=df.date, columns=vectorizer.get_feature_names())
    word_frequency_matrix = word_frequency_matrix.sort_index()
    word_frequency_matrix.to_csv(os.path.join(path_data, "word_frequency_80.csv"),
                                 sep=";", encoding="utf-8", index=False)
    return word_frequency_matrix


def build_model(model_file):
    nb_topics: int = 50
    words_per_topic: int = 20
    if os.path.exists(os.path.join(path_to_model, model_file)):
        print("il existe déjà un modèle avec ce nom")
    else:
        word_frequency_matrix = pd.read_csv(os.path.join(path_data, "word_frequency_80.csv"), sep=";", encoding="utf-8",
                                            index_col=0)
        clefs: List[str] = list(word_frequency_matrix.columns)
        print(clefs[:10])
        blocs: List[str] = list(word_frequency_matrix.index)
        print(blocs[:10])
        print("Topic modeling")
        lda = LatentDirichletAllocation(n_components=nb_topics)
        topic_to_text = lda.fit_transform(word_frequency_matrix.values)
        pkl_filename = os.path.join(path_to_model, model_file)
        with open(pkl_filename, 'wb') as file:
            pickle.dump(lda, file)
    all_topics: pd.DataFrame = pd.DataFrame({f"Topic{i}": [clefs[w] for w in top.argsort()[-words_per_topic:]]
                                             for i, top in enumerate(lda.components_)})
    table_topics_to_texts: pd.DataFrame = pd.DataFrame(np.vectorize(lambda z: f"{z:.3f}")(topic_to_text),
                                                       columns=range(nb_topics), index=blocs)
    all_topics.to_excel(os.path.join(path_data, "topics.xlsx"), encoding="utf-8", index=False)
    table_topics_to_texts.to_excel(os.path.join(path_data, "corpus_topics.xlsx"), encoding="utf-8", index=True)


def load_model(model_file):
    word_freq = pd.read_csv(os.path.join(path_data, "word_frequency_80.csv"), sep=";", encoding="utf-8", index_col=0)
    clefs: List[str] = list(word_freq.columns)
    blocs: List[str] = list(word_freq.index)
    pkl_filename = os.path.join(path_to_model, model_file)
    with open(pkl_filename, 'rb') as file:
        lda = pickle.load(file)
    nb_topics: int = lda.n_components
    words_per_topic: int = 20
    topic_to_text = lda.transform(word_freq.values)
    all_topics: pd.DataFrame = pd.DataFrame({f"Topic{i}": [clefs[w] for w in top.argsort()[-words_per_topic:]]
                                             for i, top in enumerate(lda.components_)})
    table_topics_to_texts: pd.DataFrame = pd.DataFrame(np.vectorize(lambda z: f"{z:.3f}")(topic_to_text),
                                                       columns=range(nb_topics), index=blocs)
    all_topics.to_excel(os.path.join(path_data, "topics.xlsx"), encoding="utf-8", index=False)
    table_topics_to_texts.to_excel(os.path.join(path_data, "corpus_topics.xlsx"), encoding="utf-8", index=True)
    return topic_to_text, all_topics, table_topics_to_texts


def get_parameter(adr: str):
    pkl_filename = os.path.join(path_to_model, adr)
    with open(pkl_filename, 'rb') as file:
        lda: LatentDirichletAllocation = pickle.load(file)
    print(lda.get_params())


if __name__ == "__main__":

    df = table_ocr_1880()  # collecte dans un dataframe, les textes des années 80 avec la date comme référence
    df = build_corpus(df)  # calcul les bag of words pour chaque document, puis les ajoute dans le dataframe précédent
    word_frequency = count_vectorizer(df, 10000)  # à partir des bag of words, calcul la matrice de fréquence par doc
    build_model('lda_model_blocs.pkl')  # lance un modèle de LDA. Il ne fera rien si un modèle du même nom existe déjà
    text_topics, topics, table_text_topics = load_model("lda_model_blocs.pkl")
    get_parameter("lda_model_blocs.pkl")

