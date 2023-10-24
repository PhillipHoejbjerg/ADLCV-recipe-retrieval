import os
import shutil
import kaggle

# for this to work you need to place your kaggle api key in ~/.kaggle/kaggle.json
kaggle.api.authenticate()

kaggle.api.dataset_download_files('pes12017000148/food-ingredients-and-recipe-dataset-with-images', path='data/raw/Food Images', unzip=True)

shutil.copytree('data/raw/Food Images', 'data/processed/Food Images')
os.remove('data/processed/Food Images/pan-seared-salmon-on-baby-arugula-242445.jpg')
