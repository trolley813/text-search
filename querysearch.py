from pprint import pprint

from nltk import TreebankWordTokenizer
from nltk.stem import PorterStemmer
import clickhouse_driver
from math import log2

def bm25(doc_len, avg_len, idfs, counts):
    k1 = 2.0
    b = 0.75
    return sum(idfs[i] * (counts[i] / doc_len * (k1 + 1)) / (counts[i] / doc_len + k1 * (1 - b + b * doc_len / avg_len))
               for i in range(len(idfs)) if idfs[i] > 0)


def idf(word_id):
    con = clickhouse_driver.connect("clickhouse://127.0.0.1")
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    n_total = cur.fetchone()[0] if cur.rowcount else 0
    cur.execute("SELECT * FROM idf_doc_count WHERE word_id = %(id)s", {"id": word_id})
    n_word = cur.fetchone()[1] if cur.rowcount else 0
    con.close()
    return log2((n_total - n_word + 0.5) / (n_word + 0.5)) - log2(0.5 / (n_total + 0.5))


def bi_idf(word_id_1, word_id_2):
    con = clickhouse_driver.connect("clickhouse://127.0.0.1")
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    n_total = cur.fetchone()[0] if cur.rowcount else 0
    cur.execute("SELECT * FROM idf_doc_count_bigrams WHERE word_id_1 = %(id1)s AND word_id_2 = %(id2)s",
                {"id1": word_id_1, "id2": word_id_2})
    n_word = cur.fetchone()[2] if cur.rowcount else 0
    con.close()
    return log2((n_total - n_word + 0.5) / (n_word + 0.5)) - log2(0.5 / (n_total + 0.5))


def tri_idf(word_id_1, word_id_2, word_id_3):
    con = clickhouse_driver.connect("clickhouse://127.0.0.1")
    cur = con.cursor()
    cur.execute("SELECT COUNT(*) FROM documents")
    n_total = cur.fetchone()[0] if cur.rowcount else 0
    cur.execute("SELECT * FROM idf_doc_count_trigrams WHERE word_id_1 = %(id1)s "
                "AND word_id_2 = %(id2)s AND word_id_3 = %(id3)s",
                {"id1": word_id_1, "id2": word_id_2, "id3": word_id_3})
    n_word = cur.fetchone()[2] if cur.rowcount else 0
    con.close()
    return log2((n_total - n_word + 0.5) / (n_word + 0.5)) - log2(0.5 / (n_total + 0.5))

def querysearch(query):
    weights = [1, 10, 100]
    results = []
    con = clickhouse_driver.connect("clickhouse://127.0.0.1")
    ps = PorterStemmer()
    print("Splitting words")
    ids = []
    cur = con.cursor()
    cur.execute("SELECT * FROM avg_doc_len")
    avg_len = cur.fetchone()[0]
    for word_start, word_end in TreebankWordTokenizer().span_tokenize(query):
        word = query[word_start:word_end]
        stem = ps.stem(word)
        cur = con.cursor()
        cur.execute("SELECT id FROM words WHERE word = %(word)s", {"word": stem})
        row = cur.fetchone()
        if row is None:
            print(f"Warning: Word {word} in form of {stem} not found in a database, skipping")
        else:
            id = row[0]
            ids.append(id)
    print("Querying the database")
    id_sets = []
    for id in ids:
        print(f"Trying word with id {id}")
        cur = con.cursor()
        cur.execute("SELECT DISTINCT document_id FROM doc_words WHERE word_id = %(id)s", {"id": id})
        rows = cur.fetchall()
        id_sets.append({row[0] for row in rows})
    idfs = [idf(word_id) for word_id in ids]
    bi_idfs = [bi_idf(word_id_1, word_id_2)
               for (word_id_1, word_id_2) in ((ids[i], ids[i + 1]) for i in range(len(ids) - 1))]
    tri_idfs = [tri_idf(word_id_1, word_id_2, word_id_3)
               for (word_id_1, word_id_2, word_id_3) in
               ((ids[i], ids[i + 1], ids[i + 2]) for i in range(len(ids) - 2))]
    idfs = [idf(word_id) for word_id in ids]
    print(f"IDFs: {idfs}")
    print(f"Bigram IDFs: {bi_idfs}")
    print(f"Trigram IDFs: {tri_idfs}")
    docs = id_sets[0].intersection(*id_sets[1:])
    print(f"{len(docs)} documents found")
    for doc in docs:

        cur = con.cursor()
        cur.execute("SELECT name FROM documents WHERE id = %(did)s", {"did": doc})
        doc_name = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM inv_index WHERE document_id = %(did)s", {"did": doc})
        doc_len = cur.fetchone()[0]
        # ordinary word count
        counts = []
        for id in ids:
            cur = con.cursor()
            cur.execute("SELECT word_count FROM doc_words WHERE word_id = %(wid)s AND document_id = %(did)s",
                        {"wid": id, "did": doc})
            counts.append(cur.fetchone()[0])
        # bigram count
        bi_counts = []
        for (id1, id2) in ((ids[i], ids[i+1]) for i in range(len(ids) - 1)):
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) FROM inv_index_bigrams WHERE word_id_1 = %(wid1)s AND word_id_2 = %(wid2)s "
                        "AND document_id = %(did)s",
                        {"wid1": id1, "wid2": id2, "did": doc})
            bi_counts.append(cur.fetchone()[0])
        # trigram count
        tri_counts = []
        for (id1, id2, id3) in ((ids[i], ids[i + 1], ids[i + 2]) for i in range(len(ids) - 2)):
            cur = con.cursor()
            cur.execute("SELECT COUNT(*) FROM inv_index_trigrams WHERE word_id_1 = %(wid1)s AND word_id_2 = %(wid2)s "
                        "AND word_id_3 = %(wid3)s AND document_id = %(did)s",
                        {"wid1": id1, "wid2": id2, "wid3": id3, "did": doc})
            tri_counts.append(cur.fetchone()[0])
        bm25_score = (
                weights[0] * bm25(doc_len, avg_len, idfs, counts) +
                weights[1] * bm25(doc_len - 1, avg_len - 1, bi_idfs, bi_counts) +
                weights[2] * bm25(doc_len - 2, avg_len - 2, tri_idfs, tri_counts)
        )
        print(f"Doc ID {doc}: length = {doc_len}, ordinary count = {counts}, bigram counts = {bi_counts}, "
              f"trigram counts = {tri_counts}, BM25 score = {bm25_score}")
        results.append((doc, doc_name, doc_len, counts, bi_counts, tri_counts, bm25_score))
    results.sort(key=lambda r: -r[-1])
    return results


if __name__ == "__main__":
    pprint(querysearch("Tom Sawyer"))