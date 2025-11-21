from itertools import islice
import pkg_resources
from symspellpy import SymSpell,Verbosity
import numpy as np
import math
import sionna as sn
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras import Model
import pandas as pd
import random
import torch
from transformers import BertTokenizer, BertModel

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

nltk.download('punkt')

characters = [' ',' ',' ', '!',' ', '"', '#', '$',' ', '%', '&', "'", '(', ')', ',', '-',' ', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', ']', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '½', 'Î', 'á', 'ç', 'è', 'é', 'ï', 'ð', 'ó', 'ö', 'ú', 'ü', 'ć', 'č', 'Đ', 'ę', 'Ł', 'ł', 'ń', 'Ś', 'ś', 'š', 'Ż', 'ɑ', 'ə', 'ʂ', 'ʃ', 'ˈ', 'ː', 'Н', 'Т', 'а', 'е', 'и', 'к', 'л', 'о', 'с', '–', '—', '’', '⅓']

def char_to_binary(char, characters):
    index = characters.index(char)
    binary_str = bin(index)[2:].zfill(7) 
    return binary_str

def binary_to_char(binary_str, characters):
    return characters[int(binary_str, 2)]

def encode_string(string, characters):
    encoded = ''
    for char in string:
        if char in characters:
            encoded += char_to_binary(char, characters)
    encoded_tensor = tf.convert_to_tensor(list(map(float, encoded)), dtype=tf.float32)
    return encoded_tensor

def decode_string(encoded_str, characters):
    decoded = ''
    encoded_list = encoded_str.numpy().tolist()
    encoded_str = [str(int(num)) for num in encoded_list] 
    for i in range(0, len(encoded_str), 7):
        binary_str = ''.join(encoded_str[i:i+7])
        decoded += binary_to_char(binary_str, characters)
    return decoded

def generate_map(config):
    data_ratio = 1 - config.param['p']
    map_length = config.param['map_length']
    M = config.param['M']
    
    map_lists = set()
    num_ones = int(data_ratio * map_length)
    num_zeros = map_length - num_ones
    
    while len(map_lists) < M:
        new_list = [0] * num_zeros + [1] * num_ones
        random.shuffle(new_list)
        new_list_tuple = tuple(new_list)
        map_lists.add(new_list_tuple)
    
    return [list(t) for t in map_lists]


def optimal_map(config, score_vector, map_lists):
    map_length = config.param['map_length']
    split_lists = [score_vector[i:i+map_length] for i in range(0, len(score_vector), map_length)]
    
    max_lists = []
    max_indices = []
    for part in split_lists:
        max_product = -float('inf')
        max_list = None
        max_index = -1
        for index, map_ in enumerate(map_lists):
            product = np.dot(part, map_)
        
            if product > max_product:
                max_product = product
                max_list = map_
                max_index = index
                
        max_lists.append(max_list)
        max_indices.append(max_index)
    return max_lists, max_indices

def random_map(config, score_vector, map_lists):
    map_length = config.param['map_length']
    rand_lists = []
    num = int(len(score_vector)/map_length)
    rand_indices = np.random.randint(0, len(map_lists), size=num).tolist()
    for _, map_idx in enumerate(rand_indices):
        rand_lists.append(map_lists[map_idx])
    return rand_lists, rand_indices

class Importance_score_calculator:
    def __init__(self, config):
        self.config = config
        self.sym_spell = SymSpell()
        dictionary_path = pkg_resources.resource_filename("symspellpy", "frequency_dictionary_en_82_765.txt")
        bigram_path = pkg_resources.resource_filename("symspellpy", "frequency_bigramdictionary_en_243_342.txt")
        self.sym_spell.load_dictionary(dictionary_path, 0, 1)
        
    def alphabetic_score(self, word):
        alpha = self.config.param['alpha']
        beta = self.config.param['beta']
        gamma = self.config.param['gamma']
        word = word.lower()

        importance_scores_list = [None] * len(word)  
        characters_to_replace = list(word)
        suggestions_lengths = [float('inf')] * len(word)
        
        for i, char in enumerate(characters_to_replace):
            if char != '*':  
                temp_word = characters_to_replace[:i] + ['*'] + characters_to_replace[i+1:]
                suggestions = self.sym_spell.lookup(''.join(temp_word), Verbosity.CLOSEST, max_edit_distance=1)

                words = []
                for suggestion in suggestions:
                    words.append(suggestion.term)

                if word not in words:
                    importance_scores_list[i] = 0
                suggestions_lengths[i] = len(suggestions)
                
        if suggestions_lengths:
            min_length = min(suggestions_lengths)
            min_length_indices = [index for index, length in enumerate(suggestions_lengths) if length == min_length]
            min_suggestions_index = random.choice(min_length_indices)

            if min_length == 1 and importance_scores_list[min_suggestions_index] != 0:
                importance_scores_list[min_suggestions_index] = -1 * alpha / min_length

            elif min_length > 0 and importance_scores_list[min_suggestions_index] != 0:
                importance_scores_list[min_suggestions_index] = - (beta / min_length)

            else:
                importance_scores_list[min_suggestions_index] = 0
            characters_to_replace[min_suggestions_index] = '*'

        if len(word) == 1:
            return importance_scores_list

        suggestions_lengths = [float('inf')] * len(word)
        for i, char in enumerate(characters_to_replace):
            if char != '*': 
                temp_word = characters_to_replace[:i] + ['*'] + characters_to_replace[i+1:]
                suggestions = self.sym_spell.lookup(''.join(temp_word), Verbosity.CLOSEST, max_edit_distance=2)
                suggestions_lengths[i] = len(suggestions)

                words = []
                for suggestion in suggestions:
                    words.append(suggestion.term)
                if word not in words:
                    importance_scores_list[i] = 0
                suggestions_lengths[i] = len(suggestions)

        if suggestions_lengths:

            min_length = min(suggestions_lengths)
            min_length_indices = [index for index, length in enumerate(suggestions_lengths) if length == min_length]
            min_suggestions_index = random.choice(min_length_indices)

            if min_length == 1 and importance_scores_list[min_suggestions_index] != 0:
                importance_scores_list[min_suggestions_index] = -1 * alpha / min_length

            elif min_length > 0 and importance_scores_list[min_suggestions_index] != 0:
                importance_scores_list[min_suggestions_index] = - (beta / min_length)

            else:
                importance_scores_list[min_suggestions_index] = 0

            for i, score in enumerate(importance_scores_list):
                if score == None:
                    if suggestions_lengths[i] == 0:
                        importance_scores_list[i] = 0
                    else:
                        importance_scores_list[i] = - (gamma/suggestions_lengths[i])

        return importance_scores_list
    
    def non_alphabetic_score(self, word):
        delta = self.config.param['delta']
        word = word.lower()
        word_space = word + "*"
        suggestions = self.sym_spell.lookup(word_space, Verbosity.CLOSEST, max_edit_distance=1)
        words = []
        for suggestion in suggestions:
            words.append(suggestion.term)
        length = len(suggestions)

        if word not in words:
            score = 0
        elif length == 0:
            score = 0
        else:
            score = -1 * delta/length
        return score
    
    def text_score(self, text):
        scores = []
        word = ''  

        for letter in text:
            if letter.isalpha():  
                word += letter 
            else:
                if word:  
                    scores.extend(self.alphabetic_score(word))  
                non_alphabetic = self.non_alphabetic_score(word)
                scores.append(non_alphabetic)  
                word = ''  

        if word:  
            scores.extend(self.alphabetic_score(word))

        return scores
    
def marked(text, maps):
    combined_map = [item for sublist in maps for item in sublist]
    
    if len(combined_map) > len(text):
        combined_map = combined_map[:len(text)]    
    result = ""
    
    for i, char in enumerate(text):
        if combined_map[i] == 0:
            result += "*"
        else:
            result += char
    
    return result

def padded_score(config, score_vector):
    map_length = config.param['map_length']
    alpha = config.param['alpha']
    if isinstance(score_vector, np.ndarray):
        score_vector = score_vector.tolist()
    
    remainder = len(score_vector) % map_length
    padding_size = map_length - remainder if remainder != 0 else 0
    padded_list = score_vector + [-1*alpha] * padding_size
    return padded_list

class CodedSystemAWGN(Model):  
    def __init__(self, config, mode):
        super().__init__()  
        self.config = config
        self.num_bits_per_symbol = self.config.param['bits_per_symbol']
        self.n = self.config.param['block_length']
        if mode == 0: # proposed
            self.coderate = self.config.param['code_rate']
            self.k = self.n * self.coderate
        else: # traditional
            self.k = 7 * self.config.param['map_length']
            self.coderate = self.k/self.n

        self.constellation = sn.mapping.Constellation("qam", self.num_bits_per_symbol)
        self.mapper = sn.mapping.Mapper(constellation=self.constellation)
        self.demapper = sn.mapping.Demapper("app", constellation=self.constellation)
        self.awgn_channel = sn.channel.AWGN()

        self._init_fec()
    
    def calculate_eb_no(self):
        snr_db = self.config.param['SNR']
        snr_linear = 10 ** (snr_db / 10.0)
        eb_no_linear = snr_linear / (self.coderate * self.num_bits_per_symbol)
        eb_no_db = 10 * math.log10(eb_no_linear)

        return eb_no_db

    def _init_fec(self):
        self.encoder = sn.fec.ldpc.LDPC5GEncoder(self.k, self.n)
        self.decoder = sn.fec.ldpc.LDPC5GDecoder(self.encoder, num_iter=20, hard_out=True)
        
    def awgn(self, x):
        snr_db = self.config.param['SNR']
        snr_linear = 10 ** (snr_db / 10.0)
        signal_power = np.sum(np.abs(x) ** 2) / (x.shape[0]*x.shape[1])
        
        power_noise_total = signal_power / snr_linear
        power_noise_total = tf.cast(power_noise_total,dtype=tf.float32)
        power_noise_per_component = power_noise_total / 2
        sigma = tf.sqrt(power_noise_per_component)

        noise_real = tf.random.normal(tf.shape(tf.math.real(x)), mean=0.0, stddev=sigma, dtype=tf.float32)
        noise_imag = tf.random.normal(tf.shape(tf.math.imag(x)), mean=0.0, stddev=sigma, dtype=tf.float32)
        noise = tf.complex(noise_real, noise_imag)
        y = x + noise

        return y

    def __call__(self, bits):
        snr_db = self.config.param['SNR']
        ebno_db = self.calculate_eb_no()
        no = sn.utils.ebnodb2no(ebno_db, num_bits_per_symbol=self.num_bits_per_symbol, coderate=self.coderate)

        codewords = self.encoder(bits)
        x = self.mapper(codewords)
        y = self.awgn(x)
        llr = self.demapper([y,no])
        bits_hat = self.decoder(llr)
        return bits_hat
    
class Similarity():
    def __init__(self):
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = BertModel.from_pretrained("bert-base-cased")
        
    def compute_score(self, real, predicted):
        
        encoded_real = self.tokenizer(real, return_tensors='pt')
        encoded_predicted = self.tokenizer(predicted, return_tensors='pt')
        
        encoded_real = {k: v.to(self.model.device) for k, v in encoded_real.items()}
        encoded_predicted = {k: v.to(self.model.device) for k, v in encoded_predicted.items()}

        output_real = self.model(**encoded_real)
        output_predicted = self.model(**encoded_predicted)

        embedding_real = output_real[0]
        embedding_predicted = output_predicted[0]

        b1, w1, h1 = embedding_real.shape
        b2, w2, h2 = embedding_predicted.shape

        max_length = 1000
        padding1 = torch.zeros(b1, max_length - w1, h1)
        padding2 = torch.zeros(b2, max_length - w2, h2)

        padded_real = torch.cat((embedding_real, padding1), dim=1)
        padded_predicted = torch.cat((embedding_predicted, padding2), dim=1)

        padded_real = torch.flatten(padded_real)
        padded_predicted = torch.flatten(padded_predicted)

        norm_real = torch.norm(padded_real)
        norm_predicted = torch.norm(padded_predicted)

        score = torch.dot(padded_real, padded_predicted) / (norm_real * norm_predicted)

        return score

def calculate_bleu(reference, candidate):
    reference_tokens = word_tokenize(reference)
    candidate_tokens = word_tokenize(candidate)
    smoothie = SmoothingFunction().method1
    score = sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)
    return score


def filter_sentence(maps, sentence):
    combined_map = [item for sublist in maps for item in sublist]
    result = ""
    length = min(len(combined_map), len(sentence))
    for i in range(length):
        if combined_map[i] == 1:
            result += sentence[i]
    return result

def pad_binary(config, map_index):
    length = config.param['log2M']
    binary_string = f'{map_index:0{length}b}'
    return tf.convert_to_tensor([float(bit) for bit in binary_string], dtype=tf.float32)

def idex_filtered_bit(config, filtered_sentence, map_indices):
    m = int(config.param['map_length']*(1-config.param['p']))
    binary_length = config.param['log2M']
    result = []
    for index, map_index in enumerate(map_indices):
        binary_map_index = pad_binary(config, map_index)
        segment = filtered_sentence[index*m:(index+1)*m]
        encoded_segment = encode_string(segment, characters)
        result.append(tf.concat([binary_map_index, tf.reshape(encoded_segment, [-1])], axis=0))
    final_result = tf.concat(result, axis=0)

    data_length = int(config.param['block_length']*config.param['code_rate'])
    total_length_needed = tf.size(final_result)
    remainder = total_length_needed % data_length
    if remainder != 0:
        padding_length = data_length - remainder
        final_result = tf.concat([final_result, tf.zeros(padding_length, dtype=tf.float32)], axis=0)
    return final_result

def map_selection(maps, indices):
    lists = []
    for i in range(len(indices)):
        lists.append(maps[indices[i]])
    return lists

def index_text_decoding(config, bits):
    binary_length = config.param['log2M']
    m = int(config.param['map_length']*(1-config.param['p']))
    indices = []
    texts = []
    total_bits = binary_length + 7 * m
    for i in range(0, len(bits), total_bits):
        index_bits = bits[i:i + binary_length]
        index_value = int(''.join(str(int(bit)) for bit in index_bits.numpy()), 2)
        indices.append(index_value)
        data_bits = bits[i + binary_length:i + total_bits]
        text = decode_string(data_bits, characters)
        texts.append(text)
    decoded_filtered_text = ''.join(texts)
    return indices, decoded_filtered_text

def create_marked_sentence(maps, sentence, original_length):
    combined_map = [item for sublist in maps for item in sublist]
    result = ""
    sentence_index = 0
    sentence_index = 0

    for value in combined_map:
        if sentence_index < original_length:
            if value == 1:
                if sentence_index < len(sentence):
                    result += sentence[sentence_index]
                    sentence_index += 1
                else:
                    break
            else:
                result += '*'
        else:
            break

    return result[0:original_length]

def zero_padding(config, bits):
    data_length = 7 * config.param['map_length']
    total_length_needed = tf.size(bits)
    remainder = total_length_needed % data_length
    if remainder != 0:
        padding_length = data_length - remainder
        bits = tf.concat([bits, tf.zeros(padding_length, dtype=tf.float32)], axis=0)
    return bits