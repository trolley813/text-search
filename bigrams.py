import clickhouse_driver

if __name__ == "__main__":
    con = clickhouse_driver.connect("clickhouse://127.0.0.1")
    cur = con.cursor()
    print("(Re)creating tables")
    cur.execute("DROP TABLE IF EXISTS inv_index_trigrams")
    cur.execute("DROP TABLE IF EXISTS inv_index_bigrams")
    cur.execute("""CREATE TABLE inv_index_bigrams(
                    word_id_1 INTEGER, word_id_2 INTEGER, 
                    document_id INTEGER,
                    start_pos INTEGER, end_pos INTEGER)
                    ENGINE = MergeTree()
                    ORDER BY document_id
                    PARTITION BY document_id
                    """)
    cur.execute("""CREATE TABLE inv_index_trigrams(
                        word_id_1 INTEGER, word_id_2 INTEGER, word_id_3 INTEGER, 
                        document_id INTEGER,
                        start_pos INTEGER, end_pos INTEGER)
                        ENGINE = MergeTree()
                        ORDER BY document_id
                        PARTITION BY document_id
                        """)
    cur.execute("SELECT id FROM documents ORDER BY id")
    docs = cur.fetchall()
    for doc in docs:
        print(f"Processing doc ID {doc}")
        cur.execute("""INSERT INTO inv_index_bigrams
                SELECT word_id - runningDifference(word_id) as word_id_1, 
                word_id as word_id_2, 
                document_id,
                start_pos - runningDifference(start_pos) as start_pos, 
                end_pos
                FROM inv_index WHERE document_id = %(did)s LIMIT 10000000 OFFSET 1
                """, {"did": doc})
        con.commit()
        cur.execute("""INSERT INTO inv_index_trigrams
                SELECT word_id_1 - runningDifference(word_id_1) as word_id_1, 
                word_id_1 as word_id_2,
                word_id_2 as word_id_3, 
                document_id,
                start_pos - runningDifference(start_pos) as start_pos, 
                end_pos
                FROM inv_index_bigrams WHERE document_id = %(did)s LIMIT 10000000 OFFSET 1
                """, {"did": doc})
        con.commit()

    con.close()
