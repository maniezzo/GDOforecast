import pandas as pd, numpy as np
import sqlite3
import json

# deletes all previous records from same model and same cardinality
def deleteSqLite(dbfilePath,model, nboost):
    conn = sqlite3.connect(dbfilePath)
    command = conn.cursor()
    command.execute("delete from boost where model = \"AR\" and nboost = 100")
    # Commit changes and close connection
    conn.commit()
    conn.close()

# insert a bootstrap set into database boost as nboost rows
def insertSqlite(dbfilePath,model, nboost, idseries, boost_set):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(dbfilePath)
    command = conn.cursor()

    # Create table (just to remember its schema)
    command.execute('''
    create table if not exists boost(id integer primary key autoincrement, model text, nboost int, idseries int, series text)
    ''')

    for i in range(len(boost_set)):
        # Convert array to list and then to  JSON string
        jarray = json.dumps(boost_set[i].tolist())
        # Insert into database
        command.execute('INSERT INTO boost (model, nboost, idseries, series) VALUES (?,?,?,?)', (model,nboost,i,jarray))
        conn.commit()
    conn.close()