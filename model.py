import keras  
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import TensorBoard
import numpy as np
import os
import random


model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(200, 200, 3),
                 activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(128, (3, 3), padding='same',
                 activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

learning_rate = 0.0001
opt = keras.optimizers.Adam(lr=learning_rate, decay=1e-6)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

tensorboard = TensorBoard(log_dir="logs/STAGE1")

train_data_dir = r"C:\Users\begzh\OneDrive\Desktop\sc2_ai\train_data"


def check_data():
    choices = {"no_attack": no_attack,
               "all_in": all_in,
               "expand": expand,
               "liberator_rush": liberator_rush}

    total_data = 0

    lengths = []
    for choice in choices:
        print("Length of {} is: {}".format(choice, len(choices[choice])))
        total_data += len(choices[choice])
        lengths.append(len(choices[choice]))

    print("Total data length now is:", total_data)
    return lengths


hm_epochs = 10

for i in range(hm_epochs):
    current = 0
    increment = 200
    not_maximum = True
    all_files = os.listdir(train_data_dir)
    maximum = len(all_files)
    random.shuffle(all_files)

    while not_maximum:
        print("WORKING ON {}:{}".format(current, current+increment))
        no_attack = []
        all_in = []
        expand = []
        liberator_rush = []

        for file in all_files[current:current+increment]:
            full_path = os.path.join(train_data_dir, file)
            data = np.load(full_path, allow_pickle = True)
            data = list(data)
            for d in data:
                choice = np.argmax(d[0])
                if choice == 0:
                    no_attack.append(d)
                elif choice == 1:
                    all_in.append(d)
                elif choice == 2:
                    expand.append(d)
                elif choice == 3:
                    liberator_rush.append(d)

        lengths = check_data()
        lowest_data = min(lengths)

        random.shuffle(no_attack)
        random.shuffle(all_in)
        random.shuffle(expand)
        random.shuffle(liberator_rush)

        no_attack = no_attack[:lowest_data]
        all_in = all_in[:lowest_data]
        expand = expand[:lowest_data]
        liberator_rush = liberator_rush[:lowest_data]

        check_data()

        train_data = no_attack + all_in + expand + liberator_rush

        random.shuffle(train_data)
        print(len(train_data))

        test_size = min(100, len(train_data) // 10)
        batch_size = 128

        x_train_raw = np.array([i[1] for i in train_data[:-test_size]])
        y_train = np.array([i[0] for i in train_data[:-test_size]])

        x_test_raw = np.array([i[1] for i in train_data[-test_size:]])
        y_test = np.array([i[0] for i in train_data[-test_size:]])

        x_train = x_train_raw.reshape(-1, 200, 200, 3)
        x_test = x_test_raw.reshape(-1, 200, 200, 3)


        #x_train = np.array([i[1] for i in train_data[:-test_size]])
        #y_train = np.array([i[0] for i in train_data[:-test_size]])

        #x_test = np.array([i[1] for i in train_data[-test_size:]])
        #y_test = np.array([i[0] for i in train_data[-test_size:]])


        model.fit(x_train, y_train,
                  batch_size=batch_size,
                  validation_data=(x_test, y_test),
                  shuffle=True,
                  verbose=1, callbacks=[tensorboard])

        model.save("BasicCNN-{}-epochs-{}-LR-STAGE1".format(hm_epochs, learning_rate))
        current += increment
        if current > maximum:
            not_maximum = False


# time spent : 3 months and 1 week