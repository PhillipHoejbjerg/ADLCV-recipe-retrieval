import os
import shutil
# import kaggle
import pandas as pd

# for this to work you need to place your kaggle api key in ~/.kaggle/kaggle.json
# kaggle.api.authenticate()

# kaggle.api.dataset_download_files('pes12017000148/food-ingredients-and-recipe-dataset-with-images', path='data/raw/Food Images', unzip=True)

# shutil.copytree('data/raw/Food Images', 'data/processed/Food Images')
# os.remove('data/processed/Food Images/pan-seared-salmon-on-baby-arugula-242445.jpg')

df = pd.read_csv('data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv')
print('len before', len(df))
df = df[df['Image_Name'] != 'pan-seared-salmon-on-baby-arugula-242445']
df.dropna(inplace=True)

print(f"number of invalid images: {sum(df['Image_Name'] == '#NAME?')}")
df = df[df['Image_Name'] != '#NAME?']
df.to_csv('data/Food Ingredients and Recipe Dataset with Image Name Mapping.csv', index=False)  
print('len after', len(df))