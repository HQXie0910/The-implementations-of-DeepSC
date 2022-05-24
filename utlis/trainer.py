import tensorflow as tf
from models.modules import create_masks, loss_function, create_look_ahead_mask, create_padding_mask
from models.transceiver import sample_batch, mutual_information

@tf.function
def train_step(inp, tar, net, mine_net, optim_net, optim_mi, channel='AWGN', n_std=0.1, train_with_mine=False):
    # loss, loss_mine, mi_numerical = train_step(inp, tar)
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

    with tf.GradientTape() as Tape:
        # semantic encoder
        outs = net(inp, tar_inp, channel=channel, n_std=n_std,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)

        predictions, channel_enc_output, received_channel_enc_output = outs
        loss_error = loss_function(tar_real, predictions)

        loss = loss_error

        joint, marginal = sample_batch(channel_enc_output, received_channel_enc_output)

        mi_lb, _, _ = mutual_information(joint, marginal, mine_net)
        loss_mine = -mi_lb
        if train_with_mine:
            loss = loss_error + 0.05 * loss_mine
    # compute loss gradients
    gradients = Tape.gradient(loss, net.trainable_variables)
    # updata gradients
    optim_net.apply_gradients(zip(gradients, net.trainable_variables))

    with tf.GradientTape() as Tape:
        outs = net(inp, tar_inp, channel=channel, n_std=0.1,
                   training=True, enc_padding_mask=enc_padding_mask,
                   combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)

        predictions, channel_enc_output, received_channel_enc_output = outs

        joint, marginal = sample_batch(channel_enc_output, received_channel_enc_output)

        mi_lb, _, _ = mutual_information(joint, marginal, mine_net)
        loss_mine = -mi_lb

        mi_numerical = 2.20

    # compute loss gradients
    gradients = Tape.gradient(loss_mine, mine_net.trainable_variables)
    # updata gradients
    optim_mi.apply_gradients(zip(gradients, mine_net.trainable_variables))

    return loss, loss_mine, mi_numerical

@tf.function
def eval_step(inp, tar, net, channel='AWGN', n_std=0.1):
    # loss, loss_mine, mi_numerical = train_step(inp, tar)
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one

    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
    # semantic encoder
    outs = net(inp, tar_inp, channel=channel, n_std=n_std,
               training=False, enc_padding_mask=enc_padding_mask,
               combined_mask=combined_mask, dec_padding_mask=dec_padding_mask)

    predictions, channel_enc_output, received_channel_enc_output = outs
    loss_error = loss_function(tar_real, predictions)

    return loss_error


def greedy_decode(args, inp, net, channel='AWGN', n_std=0.1):
    bs, sent_len = inp.shape
    # notice all of the test sentence add the <start> and <end>
    # using <start> as the start of decoder
    outputs = args.start_idx*tf.ones([bs,1], dtype=tf.int32)
    # strat decoding
    enc_padding_mask = create_padding_mask(inp)
    sema_enc_output = net.semantic_encoder.call(inp, False, enc_padding_mask)
    # channel encoder
    channel_enc_output = net.channel_encoder.call(sema_enc_output)
    # over the AWGN channel
    if channel == 'AWGN':
        received_channel_enc_output = net.channel_layer.awgn(channel_enc_output, n_std)
    elif channel == 'Rician':
        received_channel_enc_output = net.channel_layer.fading(channel_enc_output, 1, n_std)
    else:
        received_channel_enc_output = net.channel_layer.fading(channel_enc_output, 0, n_std)

    for i in range(args.max_length):
        # create sequence padding
        look_ahead_mask = create_look_ahead_mask(tf.shape(outputs)[1])
        dec_target_padding_mask = create_padding_mask(outputs)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        # channel decoder
        received_channel_dec_output = net.channel_decoder.call(received_channel_enc_output)
        # semantic deocder
        predictions, _ = net.semantic_decoder.call(outputs, received_channel_dec_output,
                                                   False, combined_mask, enc_padding_mask)

        # choose the word from axis = 1
        predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)

        outputs = tf.concat([outputs, predicted_id], axis=-1)

    return outputs


@tf.function
def train_mine_step(inp, tar, net, mine_net, optim_mi, channel='AWGN', n_std=0.1):
    tar_inp = tar[:, :-1]  # exclude the last one
    tar_real = tar[:, 1:]  # exclude the first one

    with tf.GradientTape() as Tape:
        # strat decoding
        enc_padding_mask = create_padding_mask(inp)
        sema_enc_output = net.semantic_encoder.call(inp, False, enc_padding_mask)
        # channel encoder
        channel_enc_output = net.channel_encoder.call(sema_enc_output)
        # over the AWGN channel
        if channel == 'AWGN':
            received_channel_enc_output = net.channel_layer.awgn(channel_enc_output, n_std)
        elif channel == 'Rician':
            received_channel_enc_output = net.channel_layer.fading(channel_enc_output, 1, n_std)
        else:
            received_channel_enc_output = net.channel_layer.fading(channel_enc_output, 0, n_std)

        joint, marginal = sample_batch(channel_enc_output, received_channel_enc_output)
        mi_lb, _, _ = mutual_information(joint, marginal, mine_net)
        loss_mine = -mi_lb

    # compute loss gradients
    gradients = Tape.gradient(loss_mine, mine_net.trainable_variables)
    # updata gradients
    optim_mi.apply_gradients(zip(gradients, mine_net.trainable_variables))

    return loss_mine