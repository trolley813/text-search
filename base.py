import os
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk import TreebankWordTokenizer
from nltk.stem import PorterStemmer
import sqlite3

if __name__ == "__main__":
    data_path = "./data/Gutenberg/txt"
    files = [f for f in os.listdir(data_path) if f.endswith(".txt")]
    file_count = len(files)
    con = sqlite3.connect("inverted_index.db")
    con.execute("CREATE TABLE words(id INTEGER PRIMARY KEY, word TEXT)")
    con.execute("CREATE TABLE documents(id INTEGER PRIMARY KEY, name TEXT)")
    con.execute("""CREATE TABLE inv_index(id INTEGER PRIMARY KEY AUTOINCREMENT,
                word_id INTEGER, document_id INTEGER,
                start_pos INTEGER, end_pos INTEGER, 
                FOREIGN KEY(word_id) REFERENCES words(id), 
                FOREIGN KEY(document_id) REFERENCES documents(id))
                """)
    con.commit()
    words_cache = {}
    for (idx, filename) in enumerate(files, 1):
        ps = PorterStemmer()
        print(f"""Processing file {idx} of {file_count} - "{filename}"...""")
        con.execute("INSERT INTO documents(id, name) VALUES(?, ?)", (idx, filename))
        con.commit()
        file_text = open(os.path.join(data_path, filename), encoding="utf-8", errors="replace").read()
        for sent_start, sent_end in PunktSentenceTokenizer().span_tokenize(file_text):
            sentence = file_text[sent_start:sent_end].replace("''", "  ").replace("``", "  ")
            for word_start, word_end in TreebankWordTokenizer().span_tokenize(sentence):
                word = sentence[word_start:word_end]
                stem = ps.stem(word)
                if stem not in words_cache:
                    words_cache[stem] = len(words_cache) + 1
                    con.execute("INSERT INTO words(id, word) VALUES(?, ?)", (words_cache[stem], stem))
                word_id = words_cache[stem]
                con.execute("INSERT INTO inv_index(word_id, document_id, start_pos, end_pos) VALUES(?, ?, ?, ?)",
                            (word_id, idx, sent_start + word_start, sent_start + word_end))
        con.commit()
    con.close()