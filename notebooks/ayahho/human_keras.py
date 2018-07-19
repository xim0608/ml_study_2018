from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import numpy as np

# 分析対象のカテゴリー
root_dir = "./wiki_crop/"
categories = ["woman", "man"] #男が１で女が0
nb_classes = len(categories)
image_size = 32

def main():
    data_train, data_test, label_train, label_test = np.load("./test.npy")
    # label_train[np.isnan(label_train)] = 2
    label_train = label_train.astype('int')
    label_test = label_test.astype('int')
    label_train = np_utils.to_categorical(label_train, nb_classes)
    label_test = np_utils.to_categorical(label_test, nb_classes)

    # # listlabel_train = list(label_train)
    # # print(listlt)
    # # print(type(listlt[0]))
    # # print(label_train)
    # # print(type(label_train[10]))
    # # print(len(label_train))
    # # print(type(label_train))
    # # データを正規化する
    # data_train  = data_train.astype("float") / 256 #astypeはキャスト
    # data_test   = data_test.astype("float") / 256
    # # label_train = label_train.astype('int')
    # # label_test = label_test.astype('int')
    # # listlt = list(label_train)
    # # print(listlt)
    # # print(type(listlt[0]))
    # # label_train = np_utils.to_categorical(label_train)
    # # for i in range(len(label_train)):
    # #     print(i)
    # #     print(np_utils.to_categorical(label_train[i]))
    # #     label_train[i] = np_utils.to_categorical(label_train[i])
    # #     i += 1
    #
    # # listlabel_train2 = [[] for i in range(len(label_train))]
    # listlabel_train2 = []
    # # print(len(label_train2))
    # # label_train = np_utils.to_categorical(label_train) #http://may46onez.hatenablog.com/entry/2016/07/14/122047
    # for i in range(len(label_train)):
    #     # print(i)
    #     # print(np_utils.to_categorical(label_train[i]))
    #     # print(type(np_utils.to_categorical(label_train[i])))
    #     if (np_utils.to_categorical(label_train[i]) == -9223372036854775808).all():
    #         listlabel_train2.append(numpy.ndarray(nan, dtype=float32))
    #     else:
    #         listlabel_train2.append(np_utils.to_categorical(label_train[i]))
    #     # print(listlabel_train2)
    #     i += 1
    # # print(listlabel_train2)
    # # print(np_utils.to_categorical(label_train[0]))
    # # print(label_train2)
    # # label_test = np_utils.to_categorical(label_test)
    # label_test[np.isnan(label_test)] = 2
    # label_test = label_test.astype('int')
    # listlabel_test = list(label_test)
    # listlabel_test2 = []
    # for i in range(len(label_test)):
    #     if (np_utils.to_categorical(label_test[i]) == -9223372036854775808).all():
    #         listlabel_test2.append(numpy.ndarray(nan, dtype=float32))
    #     else:
    #         listlabel_test2.append(np_utils.to_categorical(label_train[i]))
    #     i += 1
    #
    # label_train = np.array(listlabel_train2)
    # label_test  = np.array(listlabel_test2)
    # print(label_test)


    # モデルを訓練し評価する
    model = model_train(data_train, label_train)
    model_eval(model, data_test, label_test)

# モデルを構築
def build_model(in_shape):
    model = Sequential() #https://keras.io/ja/getting-started/sequential-model-guide/
    model.add(Conv2D(32, (3, 3),
        padding='same',#ゼロパディングして出力画像同じサイズにする
        input_shape=(32, 32, 3)))
    model.add(Activation('relu'))#活性化関数がrelu
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='binary_crossentropy',#誤差？？
        optimizer='rmsprop',
        metrics=['accuracy'])
    return model

#モデルを訓練する
def model_train(data, label):
    print(data.shape)
    es = EarlyStopping(monitor='val_acc', patience=0, verbose=0, mode='auto')
    model = build_model(data.shape)
    model.fit(data, label, batch_size=32, epochs=30, callbacks=[es])
    #モデルを保存する
    hdf5_file = "./humanmodel.hdf5"
    model.save_weights(hdf5_file)
    return model

#モデルを評価する
def model_eval(model, data, label):
    score = model.evaluate(data, label)
    print('loss=', score[0])
    print('accuracy=', score[1])

if __name__ == "__main__":
    main()
