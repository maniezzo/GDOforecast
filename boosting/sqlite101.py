import pandas as pd, numpy as np
import sqlite3
import json

# creates table boost, removes it if existing
def createSqlite(dbfilePath):
    conn = sqlite3.connect(dbfilePath)
    command = conn.cursor()

    # first removes the table
    command.execute("DROP TABLE IF EXISTS boost")

    # Create table boost:
    # model: AR, ARIMA, ... used for computing residuals
    # fback: 0 no backcasting, 1 with backcasting
    # fRep: 0 scramble, 1 extraction with repetition from residual distribution
    # nboost: number of generated series
    # idseries: id [0..51] pf the series replicated
    # idrepl: id [0..nboost-1] of a replica of series idseries
    # series the json representation of the replicated series
    command.execute(
        '''create table if not exists boost(id integer primary key autoincrement,
                                         model text,
                                         fback int,
                                         frep int,
                                         nboost int,
                                         idseries int,
                                         idrepl int,
                                         series text)'''
        )
    conn.commit()
    conn.close()

# deletes all previous records from same model and same cardinality
def deleteSqLite(dbfilePath,model, nboost):
    conn = sqlite3.connect(dbfilePath)
    command = conn.cursor()
    command.execute("delete from boost where model = \"AR\" and nboost = 100")
    # Commit changes and close connection
    conn.commit()
    conn.close()

# insert a bootstrap set into database boost as nboost rows
def insertSqlite(dbfilePath, model, fback, frep, nboost, idseries, boost_set):
    # Connect to SQLite database (or create it if it doesn't exist)
    conn = sqlite3.connect(dbfilePath)
    command = conn.cursor()

    for i in range(len(boost_set)):
        # Convert array to list and then to  JSON string
        jarray = json.dumps(boost_set[i].tolist())
        # Insert into database
        command.execute('INSERT INTO boost (model, fback, frep, nboost, idseries, idrepl, series) VALUES (?,?,?,?,?,?,?)',
                    (model, fback, frep, nboost, idseries, i, jarray))
        conn.commit()
    conn.close()