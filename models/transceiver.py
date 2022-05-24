from models.modules import Encoder, Decoder
import tensorflow as tf
import math


class Channels(tf.keras.Model):
    def __init__(self):
        super(Channels, self).__init__()

    def awgn(self, inputs, n_std=0.1):
        x = inputs
        y = x + tf.random.normal(tf.shape(x), mean=0.0, stddev=n_std)
        return y

    def fading(self, inputs, K=1, n_std=0.1, detector='MMSE'):
        x = inputs
        bs, sent_len, d_model = x.shape
        mean = math.sqrt(K / (2 * (K + 1)))
        std = math.sqrt(1 / (2 * (K + 1)))
        x = tf.reshape(x, (bs, -1, 2))
        x_real = x[:, :, 0]
        x_imag = x[:, :, 1]
        x_complex = tf.complex(x_real, x_imag)

        # create the fading factor
        h_real = tf.random.normal((1,), mean=mean, stddev=std)
        h_imag = tf.random.normal((1,), mean=mean, stddev=std)
        h_complex = tf.complex(h_real, h_imag)
        # create the noise vector
        n = tf.random.normal(tf.shape(x), mean=0.0, stddev=n_std)
        n_real = n[:, :, 0]
        n_imag = n[:, :, 1]
        n_complex = tf.complex(n_real, n_imag)
        # Transmit Signals
        y_complex = x_complex*h_complex + n_complex
        # Employ the perfect CSI here
        if detector == 'LS':
            h_complex_conj = tf.math.conj(h_complex)
            x_est_complex = y_complex * h_complex_conj / (h_complex * h_complex_conj)
        elif detector == 'MMSE':
            # MMSE Detector
            h_complex_conj = tf.math.conj(h_complex)
            a = h_complex * h_complex_conj + (n_std * n_std * 2)
            x_est_complex = y_complex * h_complex_conj / a
        else:
            raise ValueError("detector must in LS and MMSE")
        x_est_real = tf.math.real(x_est_complex)
        x_est_img = tf.math.imag(x_est_complex)

        x_est_real = tf.expand_dims(x_est_real, -1)
        x_est_img = tf.expand_dims(x_est_img, -1)

        x_est = tf.concat([x_est_real, x_est_img], axis=-1)
        x_est = tf.reshape(x_est, (bs, sent_len, -1))

        # method 1
        noise_level = n_std * tf.ones((bs, sent_len, 1))
        h_real = h_real * tf.ones((bs, sent_len, 1))
        h_imag = h_imag * tf.ones((bs, sent_len, 1))
        h = tf.concat((h_real, h_imag), axis=-1)
        out1 = tf.concat((h, x_est), -1)   # [bs, sent_len, 2 + d_model]

        # method 2
        y_complex_real = tf.math.real(y_complex)
        y_complex_img = tf.math.imag(y_complex)
        y_complex_real = tf.expand_dims(y_complex_real, -1)
        y_complex_img = tf.expand_dims(y_complex_img, -1)
        y = tf.concat([y_complex_real, y_complex_img], axis=-1)
        y = tf.reshape(y, (bs, sent_len, -1))
        out2 = tf.concat((h, y), -1)  # [bs, sent_len, 2 + d_model]


        return x_est



class Channel_Encoder(tf.keras.Model):
    def __init__(self, size1=256, size2=16):
        super(Channel_Encoder, self).__init__()

        self.dense0 = tf.keras.layers.Dense(size1, activation="relu")
        self.dense1 = tf.keras.layers.Dense(size2, activation=None)
        self.powernorm = tf.keras.layers.Lambda(lambda x: tf.divide(x, tf.sqrt(2 * tf.reduce_mean(tf.square(x)))))

    def call(self, inputs):
        outputs1 = self.dense0(inputs)
        outputs2 = self.dense1(outputs1)
        # POWER = tf.sqrt(tf.reduce_mean(tf.square(outputs2)))
        power_norm_outputs = self.powernorm(outputs2)

        return power_norm_outputs


class Channel_Decoder(tf.keras.Model):
    def __init__(self, size1, size2):
        super(Channel_Decoder, self).__init__()
        self.dense1 = tf.keras.layers.Dense(size1, activation="relu")
        self.dense2 = tf.keras.layers.Dense(size2, activation="relu")
        # size2 equals to d_model
        self.dense3 = tf.keras.layers.Dense(size1, activation=None)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, receives):
        x1 = self.dense1(receives)
        x2 = self.dense2(x1)
        x3 = self.dense3(x2)

        output = self.layernorm1(x1 + x3)
        return output



class Mine(tf.keras.Model):
    def __init__(self, hidden_size=10):
        super(Mine, self).__init__()
        randN_05 = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)
        bias_init = tf.keras.initializers.Constant(0)

        self.dense1 = tf.keras.layers.Dense(hidden_size, bias_initializer=bias_init, kernel_initializer=randN_05,
                                            activation="relu")
        self.dense2 = tf.keras.layers.Dense(hidden_size, bias_initializer=bias_init, kernel_initializer=randN_05,
                                            activation="relu")
        self.dense3 = tf.keras.layers.Dense(1, bias_initializer=bias_init, kernel_initializer=randN_05, activation=None)

    def call(self, inputs):
        output1 = self.dense1(inputs)
        output2 = self.dense2(output1)
        output = self.dense3(output2)
        #        output1 = self.dense1(inputs)
        #        output2 = self.dense2(output1)
        #        output = self.dense3(output2)
        return output


def mutual_information(joint, marginal, mine_net):
    t = mine_net(joint)
    et = tf.exp(mine_net(marginal))
    mi_lb = tf.reduce_mean(t) - tf.math.log(tf.reduce_mean(et))
    return mi_lb, t, et


def learn_mine(batch, mine_net, mine_net_optim, ma_et, ma_rate=0.01):
    # batch is a tuple of (joint, marginal)
    joint, marginal = batch
    joint = tf.cast(joint, dtype=tf.float32)
    marginal = tf.cast(marginal, dtype=tf.float32)
    mi_lb, t, et = mutual_information(joint, marginal, mine_net)
    ma_et = (1 - ma_rate) * ma_et + ma_rate * tf.reduce_mean(et)

    # unbiasing use moving average
    loss = -(tf.reduce_mean(t) - (1 / tf.reduce_mean(ma_et)) * tf.reduce_mean(et))
    # use biased estimator
    # loss = - mi_lb
    return loss, ma_et, mi_lb


def sample_batch(rec, noise):
    rec = tf.reshape(rec, shape=[-1, 1])
    # noise = noise[:, :, 2:]
    noise = tf.reshape(noise, shape=[-1, 1])
    rec_sample1, rec_sample2 = tf.split(rec, 2, 0)
    noise_sample1, noise_sample2 = tf.split(noise, 2, 0)
    joint = tf.concat([rec_sample1, noise_sample1], 1)
    marg = tf.concat([rec_sample1, noise_sample2], 1)
    return joint, marg


class Transeiver(tf.keras.Model):
    def __init__(self, args):
        super(Transeiver, self).__init__()

        # semantic encoder
        self.semantic_encoder = Encoder(args.encoder_num_layer, args.encoder_num_heads,
                                        args.encoder_d_model, args.encoder_d_ff,
                                        args.vocab_size, dropout_pro = args.encoder_dropout)

        # semantic decoder
        self.semantic_decoder = Decoder(args.decoder_num_layer, args.decoder_d_model,
                                        args.decoder_num_heads, args.decoder_d_ff,
                                        args.vocab_size, dropout_pro = args.decoder_dropout)

        # channel encoder
        self.channel_encoder = Channel_Encoder(256, 16)
        # channel decoder
        self.channel_decoder = Channel_Decoder(args.decoder_d_model, 512)

        # channels
        self.channel_layer = Channels()

    def call(self, inputs, tar_inp, channel='AWGN', n_std=0.1, training=False, enc_padding_mask=None,
             combined_mask=None, dec_padding_mask=None):

        sema_enc_output = self.semantic_encoder.call(inputs, training, enc_padding_mask)
        # channel encoder
        channel_enc_output = self.channel_encoder.call(sema_enc_output)
        # over the AWGN channel
        if channel=='AWGN':
            received_channel_enc_output = self.channel_layer.awgn(channel_enc_output, n_std)
        elif channel=='Rician':
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, 1, n_std)
        else:
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, 0, n_std)

        # channel decoder
        received_channel_dec_output = self.channel_decoder.call(received_channel_enc_output)
        # semantic deocder
        predictions, _ = self.semantic_decoder.call(tar_inp, received_channel_dec_output,
                                                    training, combined_mask, dec_padding_mask)

        return predictions, channel_enc_output, received_channel_enc_output


    def train_semcodec(self, inputs, tar_inp, training=False, enc_padding_mask=None,
             combined_mask=None, dec_padding_mask=None):
        sema_enc_output = self.semantic_encoder.call(inputs, training, enc_padding_mask)
        predictions, _ = self.semantic_decoder.call(tar_inp, sema_enc_output,
                                                    training, combined_mask, dec_padding_mask)
        return predictions

    def train_chancodec(self, sema_enc_output, channel='AWGN', n_std=0.1):
        # channel encoder
        channel_enc_output = self.channel_encoder.call(sema_enc_output)
        # over the air
        if channel == 'AWGN':
            received_channel_enc_output = self.channel_layer.awgn(channel_enc_output, n_std)
        elif channel == 'Rician':
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, 1, n_std)
        else:
            received_channel_enc_output = self.channel_layer.fading(channel_enc_output, 0, n_std)
        # channel decoder
        received_channel_dec_output = self.channel_decoder.call(received_channel_enc_output)

        return received_channel_dec_output