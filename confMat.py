import numpy as np
import matplotlib.pyplot as plt
import keras
import tensorflow_datasets as tfds
import sklearn.metrics as sk

model = keras.models.load_model("saved_model/my_model")

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return keras.backend.cast(image, "float32") / 255., label
    #return tf.cast(image, tf.float32) / 255., label


ds_test = tfds.load(
    'emnist',
    split="test",
    shuffle_files=True,
    as_supervised=True
)

predictionsArray = []

ds_subset = ds_test.take(10000)

labelArray = []
for images, labels in ds_subset:
    labelArray.append(labels.numpy())
labelArray = np.array(labelArray)

ds_subset = ds_subset.map(normalize_img)
ds_subset = ds_subset.batch(128)
ds_subset = ds_subset.cache()
ds_subset = ds_subset.prefetch(2)

pre = np.argmax(model.predict(ds_subset), axis=1)

confMat = sk.confusion_matrix(labelArray, pre, labels=np.arange(62))
plt.imshow(confMat, cmap="Blues")
plt.show()