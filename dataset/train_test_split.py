import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split

PATH_DATASET_COLOR = 'dataset/color'


def main():
    # Create the folders
    os.makedirs('dataset/test', exist_ok=True)
    os.makedirs('dataset/train', exist_ok=True)

    color_train, color_test = train_test_split(os.listdir(PATH_DATASET_COLOR), test_size=0.2, random_state=42)

    for file in tqdm(color_train):
        os.rename(f'{PATH_DATASET_COLOR}/{file}', f'dataset/train/{file}')
        
    for file in tqdm(color_test):
        os.rename(f'{PATH_DATASET_COLOR}/{file}', f'dataset/test/{file}')

if __name__ == '__main__':
    main()