import os
import zipfile

import requests

from utils import get_size, relative_path


URL_IMG = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

DATA_FOLDER_NAME = "data"

IMG_FOLDER = "PetImages"


def print_folder_stats(path, indent=0):

    print('\t' * indent + f"{os.path.basename(path):15} - size: {get_size(path)} - containing: {len(os.listdir(path))}")


def main():

    if not os.path.exists(relative_path(DATA_FOLDER_NAME)):
        os.mkdir(DATA_FOLDER_NAME)

    r = requests.get(URL_IMG, allow_redirects = True)

    img_file_name = os.path.basename(URL_IMG)

    open(relative_path(DATA_FOLDER_NAME, img_file_name), 'wb').write(r.content)

    with zipfile.ZipFile(relative_path(DATA_FOLDER_NAME, img_file_name), "r") as zip_ref:
        zip_ref.extractall(relative_path(DATA_FOLDER_NAME))

    path_img_folder = relative_path(DATA_FOLDER_NAME, IMG_FOLDER)

    print_folder_stats(path_img_folder)

    for folder in os.listdir(path_img_folder):

        print_folder_stats(os.path.join(path_img_folder, folder), indent=1)


if __name__ == "__main__":
    main()