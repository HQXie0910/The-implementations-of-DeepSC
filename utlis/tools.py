
import numpy as np
from w3lib.html import remove_tags
from nltk.translate.bleu_score import sentence_bleu
from sklearn.preprocessing import normalize
from bert4keras.backend import keras
from bert4keras.bert import build_bert_model
from bert4keras.tokenizer import Tokenizer

class SeqtoText:
    def __init__(self, vocb_dictionary, end_idx):
        self.reverse_word_map = dict(zip(vocb_dictionary.values(), vocb_dictionary.keys()))
        self.end_idx = end_idx

    def sequence_to_text(self, list_of_indices):
        # Looking up words in dictionary
        words = []
        for idx in list_of_indices:
            if idx == self.end_idx:
                break
            else:
                words.append(self.reverse_word_map.get(idx))
        words = ' '.join(words)
        return (words)


class BleuScore():
    def __init__(self, w1, w2, w3, w4):
        self.w1 = w1  # 1-gram weights
        self.w2 = w2  # 2-grams weights
        self.w3 = w3  # 3-grams weights
        self.w4 = w4  # 4-grams weights

    def compute_score(self, real, predicted):
        score1 = []
        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1).split()
            sent2 = remove_tags(sent2).split()
            score1.append(sentence_bleu([sent1], sent2, weights=(self.w1, self.w2, self.w3, self.w4)))
        return score1


def SNR_to_noise(snr):
    snr = 10 ** (snr / 10)
    noise_std = 1 / np.sqrt(2 * snr)

    return noise_std


class Similarity():
    def __init__(self, config_path, checkpoint_path, dict_path):
        self.model1 = build_bert_model(config_path, checkpoint_path, with_pool=True)
        self.model = keras.Model(inputs=self.model1.input,
                                 outputs=self.model1.get_layer('Encoder-11-FeedForward-Norm').output)
        # build tokenizer
        self.tokenizer = Tokenizer(dict_path, do_lower_case=True)

    def compute_score(self, real, predicted):
        token_ids1, segment_ids1 = [], []
        token_ids2, segment_ids2 = [], []
        score = []

        for (sent1, sent2) in zip(real, predicted):
            sent1 = remove_tags(sent1)
            sent2 = remove_tags(sent2)

            ids1, sids1 = self.tokenizer.encode(sent1)
            ids2, sids2 = self.tokenizer.encode(sent2)

            token_ids1.append(ids1)
            token_ids2.append(ids2)
            segment_ids1.append(sids1)
            segment_ids2.append(sids2)

        token_ids1 = keras.preprocessing.sequence.pad_sequences(token_ids1, maxlen=32, padding='post')
        token_ids2 = keras.preprocessing.sequence.pad_sequences(token_ids2, maxlen=32, padding='post')

        segment_ids1 = keras.preprocessing.sequence.pad_sequences(segment_ids1, maxlen=32, padding='post')
        segment_ids2 = keras.preprocessing.sequence.pad_sequences(segment_ids2, maxlen=32, padding='post')

        vector1 = self.model.predict([token_ids1, segment_ids1])
        vector2 = self.model.predict([token_ids2, segment_ids2])

        vector1 = np.sum(vector1, axis=1)
        vector2 = np.sum(vector2, axis=1)

        vector1 = normalize(vector1, axis=0, norm='max')
        vector2 = normalize(vector2, axis=0, norm='max')

        dot = np.diag(np.matmul(vector1, vector2.T))  # a*b
        a = np.diag(np.matmul(vector1, vector1.T))  # a*a
        b = np.diag(np.matmul(vector2, vector2.T))

        a = np.sqrt(a)
        b = np.sqrt(b)

        output = dot / (a * b)
        score = output.tolist()

        return score