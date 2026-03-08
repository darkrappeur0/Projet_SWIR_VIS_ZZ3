from .setup_import import *


def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='HeNormal')(conv)
    conv = BatchNormalization()(conv, training=False)
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)
    next_layer = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv) if max_pooling else conv
    return next_layer, conv


def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    up = Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding='same')(prev_layer_input)
    merge = concatenate([up, skip_layer_input], axis=3)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same',
                  kernel_initializer='HeNormal')(conv)
    return conv


def UNetCompiled(input_size=(432, 432, 1), n_filters=32, n_classes=2):
    """
    U-Net de registration a double entree :
      - input_vis : patch VIS grayscale (H, W, 1)
      - input_ir  : patch IR           (H, W, 1)
    Les deux sont concatenes en entree (H, W, 2) pour que le reseau
    puisse comparer les deux modalites et produire le flow (dx, dy).
    """
    input_vis = Input(input_size, name='input_vis')
    input_ir  = Input(input_size, name='input_ir')

    # Concatenation des deux images -> (H, W, 2)
    inputs = concatenate([input_vis, input_ir], axis=-1)

    # Encoder
    cblock1 = EncoderMiniBlock(inputs,     n_filters,    dropout_prob=0,   max_pooling=True)
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters*2,  dropout_prob=0,   max_pooling=True)
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4,  dropout_prob=0,   max_pooling=True)
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8,  dropout_prob=0.3, max_pooling=True)
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False)

    # Decoder
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters*8)
    ublock7 = DecoderMiniBlock(ublock6,    cblock3[1], n_filters*4)
    ublock8 = DecoderMiniBlock(ublock7,    cblock2[1], n_filters*2)
    ublock9 = DecoderMiniBlock(ublock8,    cblock1[1], n_filters)

    conv9  = Conv2D(n_filters, 3, activation='relu', padding='same',
                    kernel_initializer='he_normal')(ublock9)
    conv10 = Conv2D(n_classes, 1, padding='same',
                    kernel_initializer='zeros',
                    bias_initializer='zeros',
                    name='flow_output')(conv9)

    model = tf.keras.Model(inputs=[input_vis, input_ir], outputs=conv10)
    return model