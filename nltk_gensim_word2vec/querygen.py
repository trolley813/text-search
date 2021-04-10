from textgenrnn import textgenrnn
import os
import json
import random
from querysearch import get_word_ids, idf, querysearch
from multiprocessing import Pool


def train():
    data_path = "../data/Gutenberg/txt"
    files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
    file_texts = [open(os.path.join(data_path, f), encoding="utf-8", errors="replace").read() for f in files[::20]]
    large_text = "\n".join(file_texts)
    textgen = textgenrnn()
    textgen.train_new_model([large_text], single_text=True, num_epochs=2, word_level=True, max_length=5)
    textgen.save("./tgrnn/weights.hdf5")


def get_idfs(query):
    word_ids = get_word_ids(query)
    print(f"Word ID list: {word_ids}")
    return [idf(word_id) for word_id in word_ids]

def is_good(query):
    print(f"Analysing query: {query}")
    word_ids = get_word_ids(query)
    if not word_ids:
        return False
    print(f"Word ID list: {word_ids}")
    idfs = [idf(word_id) for word_id in word_ids]
    idf_threshold_13 = 13.15375  # 40%
    idf_threshold_1 = 14.15252  # 25%
    idfs.sort(reverse=True)
    print(f"Sorted IDFs: {idfs}")
    good = (idfs[round(len(idfs) / 3) - 1] > idf_threshold_13) or (idfs[0] > idf_threshold_1)
    print(f"Query is good: {'YES!' if good else 'NO'}")
    return good


def make_good_queries(n, filename=None):
    # train()
    textgen = textgenrnn(weights_path="textgenrnn_weights.hdf5",
                         vocab_path="textgenrnn_vocab.json",
                         config_path="textgenrnn_config.json")
    textgen.load("./tgrnn/weights.hdf5")
    texts = textgen.generate(return_as_list=True, max_gen_length=8, temperature=1.0, n=n)
    texts = ["".join(c if c.isalnum() else " " for c in t) for t in texts]
    original_count = len(texts)
    print("Filtering queries...")
    texts = [t for t in texts if is_good(t)]
    print(f"Filtering finished... {len(texts)} of {original_count} queries remained")
    if filename:
        f = open(filename, "w")
        for t in texts:
            f.write(f"{t}\n")

    return texts


def get_bm25_thread(x):
    id, query = x
    path = f"./out/ref_bm25_{id}.json"
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return  # do not repeat processing the same file
    res_file = open(path, "w+")
    json.dump({query: querysearch(query)}, res_file)
    res_file.write("\n")
    res_file.flush()


def get_bm25_results():
    texts = [l[:-1] for l in open("ref_queries.txt", "r")]
    print(texts)
    with Pool(6) as p:
        p.map(get_bm25_thread, enumerate(texts, 1))
    #for t in enumerate(texts, 1):
    #    get_bm25_thread(t)


def filter_relevant():
    texts = [l[:-1] for l in open("ref_queries.txt", "r")]
    results = []
    for (id, query) in enumerate(texts, 1):
        print(f"Processing query {id}: {query}")
        data = json.load(open(f"./out/ref_bm25_{id}.json"))
        values = list(data.values())[0]
        idfs = get_idfs(query)
        top10 = [[idfs] + v[2:6] for v in values[:10]]
        results.append(top10)
    json.dump(results, open("ref_relevant.json", "w"), indent=2)


if __name__ == "__main__":
    #make_good_queries(1000, "./ref_queries.txt")
    get_bm25_results()
    filter_relevant()