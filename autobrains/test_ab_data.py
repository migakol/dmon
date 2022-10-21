import os
import sqlite3

def test1():
    # definitions
    image_folder = '/Volumes/T7/Autobrains/AB_large'
    file_list = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]


if __name__ == '__main__':
    print('Start testing AutoBrains data')