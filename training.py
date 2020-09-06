import tensorflow_datasets as tfds
import keras

(ds_train, ds_test), ds_info = tfds.load(
    'emnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True
)

def normalize_img(image, label):
    # normalize and convert to float32
    return keras.backend.cast(image, "float32") / 255., label

ds_train = ds_train.map(normalize_img)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(2)

ds_test = ds_test.map(normalize_img)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(2)

model = keras.models.Sequential()

model.add(keras.layers.Conv2D(filters=16, activation="relu", input_shape=(28,28,1), kernel_size=(3,3)))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Conv2D(filters=8, activation="relu", kernel_size=(3,3)))
model.add(keras.layers.MaxPool2D(pool_size=(2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dense(62, activation='softmax'))

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(0.001),
    metrics=['accuracy']
)

model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
    batch_size=128
)

model.save('saved_model/my_model')