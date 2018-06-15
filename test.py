# -*- coding: utf-8 -*-
# python 2.7

import csv
from shutil import copyfile
from os import path, makedirs, listdir

csv_filename = 'products.csv'
image_folder = 'images'
category_column = 'Family'
destination_folder = 'Family'

def main():
    results = {}
    with open(csv_filename, 'rU', encoding="utf8") as infile:
        data_rows = csv.DictReader(infile, delimiter=',', fieldnames=['ProductId', 'Description', 'DescriptionDate', 'SeasonOriginal', 'ProductName', 'Brand', 'Gender', 'Colour', 'Family', 'Category', 'ArticlePhotoId', 'CreateDate'])
        next(data_rows)  # Skip the header
        
        for idx, row in enumerate(data_rows):
            category = row[category_column].strip().lower().replace('/', '')
            results[row['ProductId']] = category
            
    files = listdir(path.join(image_folder))
    for f in files:
        print(f)
        product_id = f.strip().split('_')[0]
        if product_id in results:
            dst_p = path.join(destination_folder, results[product_id])
            if not path.exists(dst_p):
                makedirs(dst_p)
            copyfile(path.join(image_folder, f), path.join(dst_p, f))
    

if __name__ == '__main__':
    main()