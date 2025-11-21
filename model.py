from tools import *

class Proposed:
    def __init__(self, config, data):
        self.data = data
        self.config = config
        self.num = self.config.param['num_test']
        self.importance_score_calculator = Importance_score_calculator(self.config)
    
    def data_to_bit(self, map_lists, sentence):
        ori_score_vector = self.importance_score_calculator.text_score(sentence)
        score_vector = padded_score(self.config, ori_score_vector)
        opti_map_lists, opti_map_indices = optimal_map(self.config, score_vector, map_lists)
        punctured_sentence = filter_sentence(opti_map_lists, sentence)
        bits = idex_filtered_bit(self.config, punctured_sentence, opti_map_indices)
        bits = tf.reshape(bits,[-1,int(self.config.param['block_length']*self.config.param['code_rate'])])
        return bits
    
    def rand_data_to_bit(self, map_lists, sentence):
        ori_score_vector = self.importance_score_calculator.text_score(sentence)
        score_vector = padded_score(self.config, ori_score_vector)
        rand_map_lists, rand_map_indices = random_map(self.config, score_vector, map_lists)
        punctured_sentence = filter_sentence(rand_map_lists, sentence)
        bits = idex_filtered_bit(self.config, punctured_sentence, rand_map_indices)
        bits = tf.reshape(bits,[-1,int(self.config.param['block_length']*self.config.param['code_rate'])])
        return bits
    
    def physical_channel(self, bits):
        system = CodedSystemAWGN(self.config, 0)
        decoded = system.__call__(bits)
        return decoded
    
    def marking(self, decoded_bits, map_lists, original_length):
        decoded_bits = tf.reshape(decoded_bits, [-1])
        decoded_indices, decoded_punctured = index_text_decoding(self.config, decoded_bits)
        decoded_maps = map_selection(map_lists, decoded_indices)
        marked_sentence = create_marked_sentence(decoded_maps, decoded_punctured, original_length)
        return marked_sentence
    
    def rand_marking(self, decoded_bits, map_lists, original_length):
        decoded_bits = tf.reshape(decoded_bits, [-1])
        decoded_indices, decoded_punctured = index_text_decoding(self.config, decoded_bits)
        decoded_maps = map_selection(map_lists, decoded_indices)
        marked_sentence = create_marked_sentence(decoded_maps, decoded_punctured, original_length)
        return marked_sentence
    
    def system_model(self):
        marked_sentence = []
        rand_marked_sentence = []
        for i in range(self.num):
            sentence = self.data.iloc[[i]].to_string(index=False, header=False)
            map_lists = generate_map(self.config)
            encoded_bits = self.data_to_bit(map_lists, sentence)
            encoded_bits_rand = self.rand_data_to_bit(map_lists, sentence)
            decoded_bits = self.physical_channel(encoded_bits)
            decoded_bits_rand = self.physical_channel(encoded_bits_rand)
            marked = self.marking(decoded_bits, map_lists, len(sentence))
            marked_rand = self.rand_marking(decoded_bits_rand, map_lists, len(sentence))
            marked_sentence.append(marked)
            rand_marked_sentence.append(marked_rand)
        return marked_sentence, rand_marked_sentence

class Traditional:
    def __init__(self, config, data):
        self.data = data
        self.config = config
        self.num = self.config.param['num_test']
        
    def data_to_bit(self, sentence):
        bits = encode_string(sentence, characters)
        padded = zero_padding(self.config, bits)
        padded = tf.reshape(padded, [-1, 7 * self.config.param['map_length']])
        return padded
        
    def physical_channel(self, bits):
        system = CodedSystemAWGN(self.config, 1)
        decoded = system.__call__(bits)
        return decoded
    
    def system_model(self):
        decoded_sentence = []
        for i in range(self.num):
            sentence = self.data.iloc[[i]].to_string(index=False, header=False)
            encoded_bits = self.data_to_bit(sentence)
            decoded_bits = self.physical_channel(encoded_bits)
            decoded_bits = tf.reshape(decoded_bits, [-1])
            decoded = decode_string(decoded_bits, characters)
            decoded = decoded[0:len(sentence)]
            decoded_sentence.append(decoded)
        return decoded_sentence
    

    
    
    