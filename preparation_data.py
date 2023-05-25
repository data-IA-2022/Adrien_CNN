import os
import zipfile

import requests
import tensorflow as tf
from tensorflow.image import decode_image
from tensorflow.io import read_file, write_file

from utils import get_size, relative_path

URL_IMG = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip"

DATA_FOLDER_NAME = "data"

IMG_FOLDER = "PetImages"


def main():

    if not os.path.exists(relative_path(DATA_FOLDER_NAME)):
        os.mkdir(DATA_FOLDER_NAME)

    r = requests.get(URL_IMG, allow_redirects = True)

    img_file_name = os.path.basename(URL_IMG)

    open(relative_path(DATA_FOLDER_NAME, img_file_name), 'wb').write(r.content)

    with zipfile.ZipFile(relative_path(DATA_FOLDER_NAME, img_file_name), "r") as zip_ref:
        zip_ref.extractall(relative_path(DATA_FOLDER_NAME))

    path_img_folder = relative_path(DATA_FOLDER_NAME, IMG_FOLDER)

    for folder_name in os.listdir(path_img_folder):

        folder_path = os.path.join(path_img_folder, folder_name)

        num_skipped = 0

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            is_jfif = True
            should_remove = False

            with open(img_path, "rb") as fobj:
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)

            try:
                img = read_file(img_path)
                if not tf.io.is_jpeg(img):
                    should_remove = True

                img = decode_image(img)

                if img.ndim != 3:
                    should_remove = True

            except Exception as e:
                should_remove = True

            if (not is_jfif) or should_remove:
                num_skipped += 1
                # Delete corrupted image
                os.remove(img_path)

            else:
                tmp = tf.io.encode_jpeg(img)
                write_file(img_path, tmp)

        print(f"{folder_name:15} - size: {get_size(folder_path)} - n° img: {len(os.listdir(folder_path))} - n° corrupted: {num_skipped}")


if __name__ == "__main__":
    main()
