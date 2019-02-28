import psycopg2
from auth import dbname,host,password,port,user

class Database():
    def __init__(self):
        try:
            self.connect()
        except:
            print("Failure in connection")

    def connect(self):
        self.connection = psycopg2.connect(
            "dbname='%s' user='%s' host='%s' password='%s'"
            %(dbname,user,host,password))
        self.connection.autocommit = True
        self.cursor = self.connection.cursor()
    
    def all_messages(self):
        print('Waiting for query execution')
        sql = "SELECT DISTINCT stc.sentence, snt.sentiment FROM sentences stc\
                JOIN sentiment snt ON stc.id = snt.sentenceid\
                WHERE stc.id IN (SELECT sentenceid FROM sentiment)\
                AND snt.sentiment != 0"
        self.cursor.execute(sql)
        all = [r for r in self.cursor.fetchall()]
        return all

