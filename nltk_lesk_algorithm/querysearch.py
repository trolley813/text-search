import time
from pprint import pprint
from typing import Dict, List

from nltk import TreebankWordTokenizer
from nltk.corpus.reader.wordnet import Synset
from nltk.stem import PorterStemmer
from nltk.wsd import lesk
import clickhouse_driver
from math import log2



def bm25(doc_len, avg_len, idfs, counts):
    k1 = 2.0
    b = 0.75
    return sum(idfs[i] * (counts[i] / doc_len * (k1 + 1)) / (counts[i] / doc_len + k1 * (1 - b + b * doc_len / avg_len))
               for i in range(len(idfs)) if idfs[i] > 0)


def idf(word_ids_and_similarities: Dict[int, float]):
    con = clickhouse_driver.connect("clickhouse://127.0.0.1")
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    n_total = cur.fetchone()[0] if cur.rowcount else 0
    word_ids = ",".join(str(w) for w in word_ids_and_similarities) or "0"
    cur.execute(f"SELECT * FROM idf_doc_count WHERE word_id IN ({word_ids})")
    n_word = sum(c[1] * word_ids_and_similarities.get(c[0], 0) for c in cur.fetchmany()) if cur.rowcount else 0
    con.close()
    return log2((n_total - n_word + 0.5) / (n_word + 0.5)) - log2(0.5 / (n_total + 0.5))


def bi_idf(word_ids_and_similarities_1: Dict[int, float], word_ids_and_similarities_2: Dict[int, float]):
    con = clickhouse_driver.connect("clickhouse://127.0.0.1")
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    n_total = cur.fetchone()[0] if cur.rowcount else 0
    word_ids_1 = ",".join(str(w) for w in word_ids_and_similarities_1) or "0"
    word_ids_2 = ",".join(str(w) for w in word_ids_and_similarities_2) or "0"
    cur.execute(f"SELECT * FROM idf_doc_count_bigrams WHERE word_id_1 IN ({word_ids_1}) AND word_id_2 IN ({word_ids_2})")
    n_word = sum(c[2] * word_ids_and_similarities_1.get(c[0], 0) * word_ids_and_similarities_2.get(c[1], 0)
                 for c in cur.fetchmany()) if cur.rowcount else 0
    con.close()
    return log2((n_total - n_word + 0.5) / (n_word + 0.5)) - log2(0.5 / (n_total + 0.5))


def tri_idf(word_ids_and_similarities_1: Dict[int, float],
            word_ids_and_similarities_2: Dict[int, float],
            word_ids_and_similarities_3: Dict[int, float]):
    con = clickhouse_driver.connect("clickhouse://127.0.0.1")
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    n_total = cur.fetchone()[0] if cur.rowcount else 0
    word_ids_1 = ",".join(str(w) for w in word_ids_and_similarities_1) or "0"
    word_ids_2 = ",".join(str(w) for w in word_ids_and_similarities_2) or "0"
    word_ids_3 = ",".join(str(w) for w in word_ids_and_similarities_3) or "0"
    cur.execute(f"SELECT * FROM idf_doc_count_trigrams WHERE word_id_1 IN ({word_ids_1}) "
                f"AND word_id_2 IN ({word_ids_2}) AND word_id_3 IN ({word_ids_3})")
    n_word = sum(c[3] * word_ids_and_similarities_1.get(c[0], 0) * word_ids_and_similarities_2.get(c[1], 0)
                 * word_ids_and_similarities_3.get(c[2], 0)
                 for c in cur.fetchmany()) if cur.rowcount else 0
    con.close()
    return log2((n_total - n_word + 0.5) / (n_word + 0.5)) - log2(0.5 / (n_total + 0.5))


def get_word_ids(query):
    default_similarity = 0.2
    con = clickhouse_driver.connect("clickhouse://127.0.0.1")
    ids = []
    ps = PorterStemmer()
    words = []
    for word_start, word_end in TreebankWordTokenizer().span_tokenize(query):
        word = query[word_start:word_end]
        words.append(word)
    print(words)
    for word in words:
        synset: Synset = lesk(words, word)
        if synset is not None:
            similars = synset.similar_tos()
            similarities = {s.lemmas()[0].name(): synset.wup_similarity(s) or default_similarity for s in similars}
            similarities[synset.lemmas()[0].name()] = 1.0
            similarities[word] = 1.0
        else:
            similarities = {word: 1.0}
        print(similarities)
        stems = ",".join(f"'{ps.stem(w)}'" for w in similarities)
        similarities = {ps.stem(w): similarities[w] for w in similarities}
        cur = con.cursor()
        cur.execute(f"SELECT id, word FROM words WHERE word IN ({stems})")
        rows = cur.fetchall()
        if not rows:
            print(f"Warning: Word {word} in form of {stems} not found in a database, skipping")
        else:
            d = {}
            for row in rows:
                id = row[0]
                similarity = similarities.get(row[1], 0)
                d[id] = similarity
            ids.append(d)
    return ids


def querysearch(query):
    weights = [1, 10, 100]
    results = []
    con = clickhouse_driver.connect("clickhouse://127.0.0.1")
    print("Splitting words")
    cur = con.cursor()
    cur.execute("SELECT * FROM avg_doc_len")
    avg_len = cur.fetchone()[0]
    print("Querying the database")
    ids: List[Dict[int, float]] = get_word_ids(query)
    print(ids)
    id_sets = []
    for id in ids:
        print(f"Trying word with id {id}")
        cur = con.cursor()
        id_list = ",".join(str(i) for i in id) or "0"
        cur.execute(f"SELECT DISTINCT document_id FROM doc_words WHERE word_id IN ({id_list})")
        rows = cur.fetchall()
        id_sets.append({row[0] for row in rows})
    idfs = [idf(word_id) for word_id in ids]
    bi_idfs = [bi_idf(word_id_1, word_id_2)
               for (word_id_1, word_id_2) in ((ids[i], ids[i + 1]) for i in range(len(ids) - 1))]
    tri_idfs = [tri_idf(word_id_1, word_id_2, word_id_3)
                for (word_id_1, word_id_2, word_id_3) in
                ((ids[i], ids[i + 1], ids[i + 2]) for i in range(len(ids) - 2))]
    print(f"IDFs: {idfs}")
    print(f"Bigram IDFs: {bi_idfs}")
    print(f"Trigram IDFs: {tri_idfs}")
    docs = id_sets[0].intersection(*id_sets[1:])
    print(f"{len(docs)} documents found")
    doc_count = 0
    for doc in docs:
        doc_count += 1
        print(f"Trying doc {doc} (#{doc_count})")
        try:
            cur = con.cursor()
            cur.execute("SELECT name FROM documents WHERE id = %(did)s", {"did": doc})
            doc_name = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM inv_index WHERE document_id = %(did)s", {"did": doc})
            doc_len = cur.fetchone()[0]
            # ordinary word count
            counts = []
            for id in ids:
                cur = con.cursor()
                id_joined = ",".join(str(i) for i in id)
                cur.execute(f"SELECT word_count, word_id FROM doc_words WHERE word_id IN ({id_joined}) AND document_id = %(did)s",
                            {"did": doc})
                count = sum(r[0] * id.get(r[1], 0.0) for r in cur.fetchall())
                counts.append(count)
                cur.close()
            # bigram count
            bi_counts = []
            for (id1, id2) in ((ids[i], ids[i + 1]) for i in range(len(ids) - 1)):
                cur = con.cursor()
                id1_joined = ",".join(str(i) for i in id1)
                id2_joined = ",".join(str(i) for i in id2)
                cur.execute(
                    f"SELECT COUNT(*), word_id_1, word_id_2 FROM inv_index_bigrams WHERE word_id_1 IN ({id1_joined})"
                    f" AND word_id_2 IN ({id2_joined}) "
                    "AND document_id = %(did)s GROUP BY word_id_1, word_id_2",
                    {"did": doc})
                count = sum(r[0] * id1.get(r[1], 0.0) * id2.get(r[2], 0.0) for r in cur.fetchall())
                bi_counts.append(count)
                cur.close()
            # trigram count
            tri_counts = []
            for (id1, id2, id3) in ((ids[i], ids[i + 1], ids[i + 2]) for i in range(len(ids) - 2)):
                cur = con.cursor()
                id1_joined = ",".join(str(i) for i in id1)
                id2_joined = ",".join(str(i) for i in id2)
                id3_joined = ",".join(str(i) for i in id3)
                cur.execute(
                    f"SELECT COUNT(*), word_id_1, word_id_2, word_id_3 FROM inv_index_trigrams "
                    f"WHERE word_id_1 IN ({id1_joined}) AND word_id_2 IN ({id2_joined}) "
                    f"AND word_id_3 IN ({id3_joined}) AND document_id = %(did)s "
                    f"GROUP BY word_id_1, word_id_2, word_id_3",
                    {"did": doc})
                count = sum(r[0] * id1.get(r[1], 0.0) * id2.get(r[2], 0.0) * id3.get(r[3], 0.0) for r in cur.fetchall())
                tri_counts.append(count)
                cur.close()
            bm25_score = (
                    weights[0] * bm25(doc_len, avg_len, idfs, counts) +
                    weights[1] * bm25(doc_len - 1, avg_len - 1, bi_idfs, bi_counts) +
                    weights[2] * bm25(doc_len - 2, avg_len - 2, tri_idfs, tri_counts)
            )
            print(f"Doc ID {doc}: length = {doc_len}, ordinary count = {counts}, bigram counts = {bi_counts}, "
                  f"trigram counts = {tri_counts}, BM25 score = {bm25_score}")
            results.append((doc, doc_name, doc_len, counts, bi_counts, tri_counts, bm25_score))
        except EOFError:
            print("Warning: Problems with database")
    con.close()
    results.sort(key=lambda r: -r[-1])
    return results


if __name__ == "__main__":
    pprint(querysearch("Tom Sawyer"))
