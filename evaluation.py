
"This is used to evaluate the DeepSC model"
import json
import numpy as np
import tensorflow as tf
from models.transceiver import Transeiver, Mine
from utlis.tools import SeqtoText, BleuScore, SNR_to_noise, Similarity
from utlis.trainer import greedy_decode, train_mine_step
from dataset.dataloader import return_loader
from parameters import para_config

if __name__=='__main__':
    # Set random seed
    tf.random.set_seed(5)
    # choose performance metrics
    test_metrics = True
    test_bleu = True
    test_sentence_sim = False
    test_mi = False
    runs = 10
    SNR = [6]
    # Set Parameters
    args = para_config()
    # Load the vocab
    vocab = json.load(open(args.vocab_path, 'rb'))
    args.vocab_size = len(vocab['token_to_idx'])
    token_to_idx = vocab['token_to_idx']
    args.pad_idx = token_to_idx["<PAD>"]
    args.start_idx = token_to_idx["<START>"]
    args.end_idx = token_to_idx["<END>"]
    StoT = SeqtoText(token_to_idx, args.end_idx)
    # Load dataset
    train_dataset, test_dataset = return_loader(args)
    # Define the model
    mine_net = Mine()
    net = Transeiver(args)
    # Load the model from the checkpoint path
    checkpoints = tf.train.Checkpoint(Transceiver=net)
    a = tf.train.latest_checkpoint(args.checkpoint_path)
    checkpoints.restore(a)
    if test_mi:
        # learning rate
        optim_mi = tf.keras.optimizers.Adam(lr=0.001)
        for snr in SNR:
            n_std = SNR_to_noise(snr)
            for (batch, (inp, tar)) in enumerate(test_dataset):
                loss_mine = train_mine_step(inp, tar, net, mine_net, optim_mi, args.channel, n_std)
            print("SNR %f loss mine %f" % (snr, loss_mine.numpy()))

    if test_metrics:
        if test_sentence_sim:
            metrics = Similarity(args.bert_config_path, args.bert_checkpoint_path, args.bert_dict_path)
        elif test_bleu:
            metrics = BleuScore(1, 0, 0, 0)
        else:
            raise Exception('Must choose bleu score or sentence similarity')
        # Start the evaluation
        # for snr in SNR:
        n_std = SNR_to_noise(args.test_snr)
        word, target_word = [], []
        score = 0
        for run in range(runs):
            for (batch, (inp, tar)) in enumerate(test_dataset):
                preds = greedy_decode(args, inp, net, args.channel, n_std)
                sentences = preds.cpu().numpy().tolist()
                result_string = list(map(StoT.sequence_to_text, sentences))
                word = word + result_string

                target_sent = tar.cpu().numpy().tolist()
                result_string = list(map(StoT.sequence_to_text, target_sent))
                target_word = target_word + result_string

            score1 = metrics.compute_score(word, target_word)
            score1 = np.array(score1)
            score1 = np.mean(score1)
            score += score1
            print(
                'Run: {}; Type: VAL; BLEU Score: {:.5f}'.format(
                    run, score1
                )
            )
        score = score/runs
        print(
            'SNR: {}; Type: VAL; BLEU Score: {:.5f}'.format(
                args.test_snr, score
            )
        )
