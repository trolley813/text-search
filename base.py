import os
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk import TreebankWordTokenizer
from nltk.stem import PorterStemmer
import clickhouse_driver

if __name__ == "__main__":
    data_path = "./data/Gutenberg/txt"
    files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
    file_count = len(files)
    con = clickhouse_driver.connect("clickhouse://127.0.0.1")
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS inv_index")
    cur.execute("DROP TABLE IF EXISTS documents")
    cur.execute("DROP TABLE IF EXISTS words")
    cur.execute("CREATE TABLE words(id INTEGER, word TEXT) ENGINE = MergeTree() ORDER BY id")
    cur.execute("CREATE TABLE documents(id INTEGER, name TEXT) ENGINE = MergeTree() ORDER BY id")
    cur.execute("""CREATE TABLE inv_index(
                word_id INTEGER, document_id INTEGER,
                start_pos INTEGER, end_pos INTEGER)
                ENGINE = MergeTree()
                ORDER BY document_id
                PARTITION BY document_id
                """)
    con.commit()
    words_cache = {}
    for (idx, filename) in enumerate(files, 1):
        ps = PorterStemmer()
        print(f"""Processing file {idx} of {file_count} - "{filename}"...""")
        cur.executemany("INSERT INTO documents(id, name) VALUES", [(idx, filename)])
        con.commit()
        file_text = open(os.path.join(data_path, filename), encoding="utf-8", errors="replace").read()
        indices = []
        for sent_start, sent_end in PunktSentenceTokenizer().span_tokenize(file_text):
            sentence = file_text[sent_start:sent_end].replace("''", "  ").replace("``", "  ")
            for word_start, word_end in TreebankWordTokenizer().span_tokenize(sentence):
                word = sentence[word_start:word_end]
                stem = ps.stem(word)
                if stem not in words_cache:
                    words_cache[stem] = len(words_cache) + 1
                    #print(f"Adding new word {(words_cache[stem], stem)}")
                    #cur.executemany("INSERT INTO words(id, word) VALUES", [(words_cache[stem], stem)])
                word_id = words_cache[stem]
                indices.append((word_id, idx, sent_start + word_start, sent_start + word_end))
        cur.executemany("INSERT INTO inv_index(word_id, document_id, start_pos, end_pos) VALUES", indices)
        con.commit()
    cur.executemany("INSERT INTO words(id, word) VALUES", [(words_cache[stem], stem) for stem in words_cache])
    con.commit()
    cur.execute("CREATE MATERIALIZED VIEW avg_doc_len ENGINE = Log POPULATE "
                "AS SELECT AVG(*) FROM (SELECT COUNT(*) FROM inv_index GROUP BY document_id)")
    cur.execute("CREATE MATERIALIZED VIEW idf_doc_count ENGINE = Log() POPULATE "
                "AS SELECT word_id, COUNT(DISTINCT document_id) FROM inv_index GROUP BY word_id")
    cur.execute("CREATE MATERIALIZED VIEW doc_words ENGINE = MergeTree() ORDER BY word_id "
                "PARTITION BY word_id % 100 POPULATE AS "
                "SELECT word_id, document_id, count(*) AS word_count FROM inv_index GROUP BY word_id, document_id")
    con.close()