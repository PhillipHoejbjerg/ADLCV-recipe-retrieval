import os
import shutil
shutil.copy('data/raw/Food Images', 'data/processed/Food Images')
os.remove('data/processed/Food Images/pan-seared-salmon-on-baby-arugula-242445.jpg')