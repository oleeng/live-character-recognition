import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

model = tf.keras.models.load_model('saved_model/my_model')

def normalize_img(image, label):
    # normalize and convert to float32
    return tf.cast(image, tf.float32) / 255., label


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

ds_subset = ds_subset.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_subset = ds_subset.batch(128)
ds_subset = ds_subset.cache()
ds_subset = ds_subset.prefetch(tf.data.experimental.AUTOTUNE)

pre = np.argmax(model.predict(ds_subset), axis=1)

confMat = tf.math.confusion_matrix(labelArray, pre, 62)
plt.imshow(confMat, cmap="Blues")
plt.show()