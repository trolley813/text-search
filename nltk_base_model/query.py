from nltk import PorterStemmer
from nltk.tokenize import word_tokenize


def query_results(query):
    tokens = word_tokenize(query)
    stemmer = PorterStemmer()
    stems = [stemmer.stem(t) for t in tokens]
    print(stems)
    # TODO: Database search
    # Select word IDs
    word_ids = ...



if __name__ == "__main__":
    query_results("little green people")