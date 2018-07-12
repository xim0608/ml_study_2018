import sys, os, re, time, scipy
import urllib.request as req
import urllib.parse as parse
import json
import numpy as np
from scipy.io import loadmat
import argparse
from datetime import datetime
from tqdm import tqdm
from cv2 import cv2
# ここから先輩が足してた
from keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split


def calc_age(taken, dob):
    birth = datetime.fromordinal(max(int(dob) - 366, 1))

    # assume the photo was taken in the middle of the year
    if birth.month < 7:
        return taken - birth.year
    else:
        return taken - birth.year - 1

# .matの表示
# # meta = scipy.io.loadmat("wiki_crop/wiki.mat")
# meta = scipy.io.loadmat("imdb_crop/imdb.mat")
# metap = str(meta)
# # metap2 = metap[1:50000]
# print(metap)
def get_meta(mat_path, db):
    print(db)
    meta = loadmat(mat_path)
    print(meta)
    full_path = meta[db][0, 0]["full_path"][0]
    dob = meta[db][0, 0]["dob"][0]  # Matlab serial date number
    gender = meta[db][0, 0]["gender"][0]
    photo_taken = meta[db][0, 0]["photo_taken"][0]  # year
    face_score = meta[db][0, 0]["face_score"][0]
    second_face_score = meta[db][0, 0]["second_face_score"][0]
    age = [calc_age(photo_taken[i], dob[i]) for i in range(len(dob))]

    return full_path, dob, gender, photo_taken, face_score, second_face_score, age


def get_args():
    parser = argparse.ArgumentParser(description="This script cleans-up noisy labels "
                                                 "and creates database for training.",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", "-o", type=str, required=True, default="wiki_crop/wiki.mat",
                        help="path to output database mat file")
    parser.add_argument("--db", type=str, default="wiki",
                        help="dataset; wiki or imdb")
    parser.add_argument("--img_size", type=int, default=32,
                        help="output image size")
    parser.add_argument("--min_score", type=float, default=1.0,
                        help="minimum face_score")
    args = parser.parse_args()
    print("bb")
# print (args.test)
    return args

def main():
    args = get_args()
    print("aa")
    # print(args)
    output_path = args.output
    # print(output_path)
    db = args.db
    # print(db)
    img_size = args.img_size
    print(img_size)
    # print(img_size)
    min_score = args.min_score
    # print(min_score)

    root_path = "{}_crop/".format(db)
    # print(root_path)
    mat_path = root_path + "{}3.mat".format(db)
    full_path, dob, gender, photo_taken, face_score, second_face_score, age = get_meta(mat_path, db)
    # print(gender)
    print(full_path[0])
    print(gender[0])
    data = [] #ここからまるたく
    label = []
    for path, g in zip(full_path, gender):
        print('Load image. image path: {}'.format(path))
        if np.isnan(g):
            print('Pass data because nan.  full path: {}'.format(path))
            continue
        img = load_img('wiki_crop/{}'.format(path[0]), target_size=(32, 32))
        img_array = img_to_array(img)
        data.append(img_array)
        label.append(g)
        print('Load image {} finish'.format(path))
    data_npy = np.array(data)
    label_npy = np.array(label)
    data_train, data_test, label_train, label_test = train_test_split(data_npy, label_npy)
    npy_array = (data_train, data_test, label_train, label_test)
    np.save('test.npy', npy_array)
    # #ここまで先輩
    #
    # out_genders = []
    # out_ages = []
    # out_imgs = []
    #
    # for i in tqdm(range(len(face_score))):
    #     if face_score[i] < min_score:
    #         continue
    #
    #     if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
    #         continue
    #
    #     if ~(0 <= age[i] <= 100):
    #         continue
    #
    #     if np.isnan(gender[i]):
    #         continue
    #
    #     out_genders.append(int(gender[i]))
    #     out_ages.append(age[i])
    #     img = cv2.imread(root_path + str(full_path[i][0]))
    #     out_imgs.append(cv2.resize(img, (img_size, img_size)))
    #
    # output = {"image": np.array(out_imgs), "gender": np.array(out_genders), "age": np.array(out_ages),
    #           "db": db, "img_size": img_size, "min_score": min_score}
    # scipy.io.savemat(output_path, output)




if __name__ == '__main__':
    main()
