import tensorflow as tf

def scheduler(epoch, lr):
    if epoch < 13:
        return lr
    elif epoch < 17:
        return lr/10
    else:
        return lr/100

def cbr(x, out_layer, kernel, stride, dilation):
    x = tf.keras.layers.Conv1D(out_layer, kernel_size=kernel, dilation_rate=dilation, strides=stride, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    return x

def wave_block(x, filters, kernel_size, n):
    dilation_rates = [2 ** i for i in range(n)]
    x = tf.keras.layers.Conv1D(filters=filters,
               kernel_size=1,
               padding='same')(x)
    res_x = x
    for dilation_rate in dilation_rates:
        tanh_out = tf.keras.layers.Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='tanh',
                          dilation_rate=dilation_rate)(x)
        sigm_out = tf.keras.layers.Conv1D(filters=filters,
                          kernel_size=kernel_size,
                          padding='same',
                          activation='sigmoid',
                          dilation_rate=dilation_rate)(x)
        x = tf.keras.layers.Multiply()([tanh_out, sigm_out])
        x = tf.keras.layers.Conv1D(filters=filters,
                   kernel_size=1,
                   padding='same')(x)
        res_x = tf.keras.layers.Add()([res_x, x])
    return res_x

def cnn(timesteps, nions, kernel_width=3, input_smoothing=4):
    abundance_in = tf.keras.layers.Input(shape=(timesteps, nions))
    temp_in = tf.keras.layers.Input(shape=(timesteps, 1))

    x_in = tf.keras.layers.concatenate([abundance_in, temp_in], axis=2)
    x_in = tf.keras.layers.Conv1D(128, input_smoothing, strides=input_smoothing)(x_in)

    x = cbr(x_in, 64, kernel_width, 1, 1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = wave_block(x, 16, kernel_width, 12)
    x = tf.keras.layers.BatchNormalization()(x)
    x = wave_block(x, 32, kernel_width, 8)
    x = tf.keras.layers.BatchNormalization()(x)
    x = wave_block(x, 64, kernel_width, 4)
    x = tf.keras.layers.BatchNormalization()(x)
    x = wave_block(x, 128, kernel_width, 1)
    x = cbr(x, 32, kernel_width, 1, 1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = wave_block(x, 64, kernel_width, 1)
    x = cbr(x, 32, kernel_width, 1, 1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.6)(x)

    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)

    out = tf.keras.layers.Dense(10, activation='sigmoid')(x)
    model = tf.keras.models.Model(inputs=[abundance_in, temp_in], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model
