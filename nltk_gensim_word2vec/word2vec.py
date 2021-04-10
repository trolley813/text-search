import os

from gensim.models import Word2Vec
from nltk import PunktSentenceTokenizer, TreebankWordTokenizer


def train():
    data_path = "../data/Gutenberg/txt"
    files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
    file_texts = [open(os.path.join(data_path, f), encoding="utf-8", errors="replace").read() for f in files[::20]]
    large_text = "\n".join(file_texts)
    sentences = []
    sent_count = 0
    for sent_start, sent_end in PunktSentenceTokenizer().span_tokenize(large_text):
        sentence_text = large_text[sent_start:sent_end].replace("''", "  ").replace("``", "  ")
        sentence = []
        for word_start, word_end in TreebankWordTokenizer().span_tokenize(sentence_text):
            word = sentence_text[word_start:word_end]
            sentence.append(word)
        sentences.append(sentence)
        sent_count += 1
        if sent_count % 1000 == 0:
            print(f"{sent_count} sentences processed...")
    print(f"Total sentences: {len(sentences)}")
    model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=16)
    model.save("word2vec.model")


if __name__ == "__main__":
    model = Word2Vec.load("word2vec.model")
    print(model.wv.most_similar("Helen", topn=13))
