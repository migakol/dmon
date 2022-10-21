import os
import pickle
import sqlite3
import pandas as pd
import csv


def save_file_list(input_folder, output_file):
    """
    The function saves running time on disk-on-key. It just creates a list of images on a disk on key folder and
    saves it into a file on harddisk
    Args:
        input_folder:
        output_file:

    Returns:

    """
    file_list = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
    with open(output_file, 'wb') as file:
        # A new file will be created
        pickle.dump(file_list, file)
    # with open('/Users/michaelko/Data/AB/diskonkeyimages.pkl', 'rb') as file:
    #     myvar = pickle.load(file)


def convert_csv_into_sqlite(csv_name):
    con = sqlite3.connect("db.sqlite")
    cursor = con.cursor()
    with open(csv_name, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=';', quotechar='|')
        cnt = 0
        for row in spamreader:
            if cnt % 1000 == 0:
                print(', '.join(row))
            sqlite_insert_blob_query = """INSERT INTO auto_brains (id, filename,pos1, pos2, signature) VALUES (?, ?, ?, ?, ?)"""
            data_tuple = (cnt, row[0], row[1], row[2], row[3])
            cursor.execute(sqlite_insert_blob_query, data_tuple)
            con.commit()
            cnt = cnt + 1
        cursor.close()

    # df = pd.read_csv(csv_name)
    pass

def create_table():
    con = sqlite3.connect("db.sqlite")
    cursor = con.cursor()
    cursor.execute('CREATE TABLE auto_brains (id INTEGER PRIMARY KEY, filename TEXT, pos1 INTEGER, pos2 INTEGER, signature TEXT);')
    con.commit()
    cursor.close()

if __name__ == '__main__':
    print('Start')
    # save_file_list('/Volumes/T7/Autobrains/AB_large', '/Users/michaelko/Data/AB/diskonkeyimages.pkl')
    convert_csv_into_sqlite('/Users/michaelko/Data/AB/large_8mp_fixed.csv')