import cv2 as c 
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score,f1_score

def data_preprocessing(Closed_door_frame_path, Open_door_frame_path):
    #write docstring
    """ 
    This function is used to preprocess the images to be fed into the model
    """
    open_door_img = []
    closed_door_img = []

    #read closed door image
    for i in range(0, 45):
        img = c.imread(Closed_door_frame_path + str(i) + ".jpg")
        img = c.resize(img, (224, 224))
        closed_door_img.append(img)

    closed_door_label = np.zeros(len(closed_door_img), dtype=int)   

    #read open door image
    for i in range(45, 130):
        img = c.imread(Open_door_frame_path + str(i) + ".jpg")
        img = c.resize(img, (224, 224)) #standard CNN format
        open_door_img.append(img)

    open_door_label = np.ones(len(open_door_img),dtype=int)

    images = np.array(closed_door_img + open_door_img) 
    labels = np.array(list(closed_door_label) + list(open_door_label))

    return images, labels


Closed_door_frame_path = "data/FramesExtracted/ClosedDoor/" 
Open_door_frame_path = "data/FramesExtracted/OpenDoor/" 

images, labels = data_preprocessing(Closed_door_frame_path, Open_door_frame_path)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = 0)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs = 10)
model.save('model/second_model.h5')

# Load the model
loaded_model = load_model('model/second_model.h5')

y_train_pred = loaded_model.predict(X_train)
y_test_pred = loaded_model.predict(X_test)

# plot roc_auc curve to evaluate model performance

print("Train ROC AUC score: ", roc_auc_score(y_train, y_train_pred)) # 0.99roc_auc_score(y_test, y_test_pred)
print("Train F1 score: ", f1_score(y_train, y_train_pred.round())) # 0.99f1_score(y_test, y_test_pred)

print("Test ROC AUC score: ", roc_auc_score(y_test, y_test_pred)) # 0.99roc_auc_score(y_test, y_test_pred
print("Test F1 score: ", f1_score(y_test, y_test_pred.round())) # 0.99f1_score(y_test, y_test_pred)
