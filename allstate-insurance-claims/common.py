import tensorflow as tf
from keras.layers import (
        CategoryEncoding,
        IntegerLookup,
        Normalization,
        StringLookup)

def df_to_dataset(dataframe, target_label, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop(target_label)
    df = {key: value[:,tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(batch_size)
    return ds

@tf.autograph.experimental.do_not_convert
def get_categorical_encoded_layer(name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        index = StringLookup(max_tokens=max_tokens)
    else:
        index = IntegerLookup(max_tokens=max_tokens)
    feature_ds = dataset.map(lambda x, y: x[name], num_parallel_calls=tf.data.AUTOTUNE)
    index.adapt(feature_ds)
    encoder = CategoryEncoding(num_tokens=index.vocabulary_size())
    return lambda feature: encoder(index(feature))

@tf.autograph.experimental.do_not_convert
def get_normalized_layer(name, dataset):
    normalizer = Normalization(axis=None)
    feature_ds = dataset.map(lambda x, y: x[name], num_parallel_calls=tf.data.AUTOTUNE)
    normalizer.adapt(feature_ds)
    return normalizer

def get_encoded_layer(column_type,
                      column_name,
                      dataset,
                      dtype):
    if 'cat' in column_type:
        return get_categorical_encoded_layer(column_name,
                                             dataset,
                                             dtype)
    elif 'cont' in column_type:
        return get_normalized_layer(column_name,
                                    dataset)
