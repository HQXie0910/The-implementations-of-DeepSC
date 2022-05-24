# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:40:09 2019

@author: HQ_Xie
This file includes the module I need
The encoder includes two sublayers
1. multiheads attention
2. Feed Forward

The first sublayer includes:
    1. Positional encoding
    2. Scaled_dot_product_attention 
    3. Multi-head attention

The second sublayer includes:
    1.Point wise feed forward network

The decoder includes three sublayers
"""
import numpy as np
import tensorflow as tf



def postional_encoder(position, d_model):
    '''
    Position encoder layer
    2i-th: sin(pos/10000^(2i/d_model))
    2i+1-th: cos(pos/10000^(2i/d_model))
    '''
    def get_angles(pos, i, d_model):
        angle_rates = pos / np.power(10000, ((2 * i) / np.float32(d_model)))
        # angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return  angle_rates
    angle_set = get_angles(np.arange(position)[:, None],
                           np.arange(d_model)[None, :],
                           d_model)
    # build odd and even index
    angle_set[:, 0::2] = np.sin(angle_set[:, 0::2]) #2i
    angle_set[:, 1::2] = np.cos(angle_set[:, 1::2]) #2i+1
    
    pos_encoding = angle_set[None, ...]
    
    return tf.cast(pos_encoding, dtype = tf.float32)


class sublayer1(tf.keras.layers.Layer):
    '''
    This is multihead function, in order to 
    '''
    def __init__(self, d_model, num_heads):
        super(sublayer1, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads        
        self.depth = d_model//num_heads
        
        assert d_model % self.num_heads == 0
        
        self.wq = tf.keras.layers.Dense(self.d_model, use_bias=False)    
        self.wk = tf.keras.layers.Dense(self.d_model, use_bias=False)
        self.wv = tf.keras.layers.Dense(self.d_model, use_bias=False)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def scale_dot_product_attention(self, q, k, v, mask):
        '''
        softmax(Q*K^t/\sqrt(d_k))*V
        where the dimension of Q is [, seq_len_q, depth]
        the dimension of K is [, seq_len_k, depth]
        the dimension of V is  [, seq_len_v, depth_v]
        Q*K^t = [, seq_len_q, seq_len_k]
    
        Notice seq_len_k = seq_len_v
    
        mask uses in the Q*K^T ...[, seq_len_q, seq_len_k]
    
        the output is [, seq_len_q, depth_v]
        '''
        # Matmul and Scale
        
        matmul_qk = tf.matmul(q, k, transpose_b = True) 
        dk = tf.cast(tf.shape(k)[-1], dtype = tf.float32)
        scaled_dot_logits = matmul_qk/tf.math.sqrt(dk)
        # where #[batch_size, seq_len_q, seq_len_k]
        
        # Mask
        # put the padding into -inf, so the effect will be eliminated in the softmax
        if mask is not None:
            scaled_dot_logits += (mask * -1e9) 
        
        
        # Softmax
        attention_weights = tf.nn.softmax(scaled_dot_logits, axis = -1)
        # where the dimension is [batch_size, seq_len_q, seq_len_k]
    
        #Output
        outputs = tf.matmul(attention_weights, v)
    
        return outputs, attention_weights
    
    def split_heads(self, x):
        '''
        divide the Q, K, V into multiheads
        (batch_size, num_heads, seq_len, depth)
        '''
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[1]

        # Calculate depth of last dimension after it has been split.
        depth = (self.d_model // self.num_heads)

        # Split the last dimension
        x = tf.reshape(x, [batch_size, length, self.num_heads, depth])

        # Transpose the result
        return tf.transpose(x, [0, 2, 1, 3])
    
    def combined_heads(self, x):
        '''
        combined multi heads
        '''
        batch_size = tf.shape(x)[0]
        length = tf.shape(x)[2]
        x = tf.transpose(x, [0, 2, 1, 3])  # --> [batch, length, num_heads, depth]
        return tf.reshape(x, [batch_size, length, self.d_model])
        
    def call(self, v, k, q, mask):
        # batch_size = tf.shape(q)[0]
        # attention_weights does not have concat
        multi_outputs = None
        
                 
        q = self.wq(q) # (batch_size, seq_len_q, d_model)
        k = self.wk(k) # (batch_size, seq_len_k, d_model)
        v = self.wv(v) # (batch_size, seq_len_v, d_model)
        
        q = self.split_heads(q) # (batch_size, num_heads, seq_len_q, depth) where d_model = num_heads*depth
        k = self.split_heads(k) # (batch_size, num_heads, seq_len_k, depth) where d_model = num_heads*depth
        v = self.split_heads(v) # (batch_size, num_heads, seq_len_v, depth) where d_model = num_heads*depth
        
        scaled_attention, attention_weights = self.scale_dot_product_attention(q, k, v, mask)
        #[batch_size, seq_len_q, depth_v]
        attention_output = self.combined_heads(scaled_attention)
                
        multi_outputs = self.dense(attention_output)
        
        return multi_outputs, attention_weights

class sublayer2(tf.keras.layers.Layer):
    def __init__(self, d_model, dff):
        super(sublayer2, self).__init__()
        self.d_model = d_model
        self.dff = dff

    def point_wise_feed_forward_network(d_model, dff):
        '''
        This is point_wise_feed_forward_network
        FFN = max(0, x*W_1 + b_1)*W_2 +b_2
        '''
        return tf.keras.Sequential([tf.keras.layers.Dense(dff, activation = 'relu'),
                                tf.keras.layers.Dense(d_model)])
    
        
        
class EncoderLayer(tf.keras.layers.Layer):
    '''
    This is encoder layer, which includes two sublayers, multihead and feed forward.
    '''
    def __init__(self, d_model, num_heads, dff, drop_pro = 0.1):
        super(EncoderLayer, self).__init__()
        
        self.sl1 = sublayer1(d_model, num_heads)
        self.sl2 = sublayer2(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(drop_pro)
        self.dropout2 = tf.keras.layers.Dropout(drop_pro)
        
    def call(self, x, training, mask):
        # attention: the layernorm(x + sublayer(x)) should be replaced by 
        # x + sublayer(LayerNorm(x)) 
        attn_output, _ = self.sl1(x, x, x, mask) #这个地方有点问题
        attn_output = self.dropout1(attn_output, training = training)
        output1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)
        
        ffn_output = self.sl2(output1)
        ffn_output = self.dropout2(ffn_output, training = training)
        output2 = self.layernorm2(output1 + ffn_output) # (batch_size, input_seq_len, d_model)
        
        return output2
    
class DecoderLayer(tf.keras.layers.Layer):
    '''
    This is decoder leayer, which includes three layers, 
    1. multihead, 
    2. masked multihead 
    3. feed forward
    '''
    def __init__(self, d_model, num_heads, dff, drop_pro = 0.1):
        super(DecoderLayer, self).__init__()
        
        self.sl11 = sublayer1(d_model, num_heads) #masked
        self.sl12 = sublayer1(d_model, num_heads)
        
        self.ffn = sublayer2(d_model, dff)
        
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = tf.keras.layers.Dropout(drop_pro)
        self.dropout2 = tf.keras.layers.Dropout(drop_pro)
        self.dropout3 = tf.keras.layers.Dropout(drop_pro)
        
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1, attn_weights1 = self.sl11(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training = training)
        output1 = self.layernorm1(x + attn1)
        
        attn2, attn_weights2 = self.sl12(enc_output, enc_output, output1, padding_mask)
        attn2 = self.dropout2(attn2, training = training)
        output2 = self.layernorm2(attn2 + output1)
        
        ffn_output = self.ffn(output2)
        ffn_output = self.dropout3(ffn_output, training = training)
        output3 = self.layernorm3(ffn_output + output2)  # (batch_size, target_seq_len, d_model)
        
        return output3, attn_weights1, attn_weights2
        

class Encoder(tf.keras.Model):
    '''
    1. Input Embedding 
    2. Positional Encoding 
    3. N encoder layers
    '''
    def __init__(self, num_layers, num_heads, d_model, dff, input_vocab_size, 
                 maximum_position_encoding=512, dropout_pro=0.1):
        super(Encoder, self).__init__()
    
        self.d_model = d_model
        self.dff = dff
        self.num_layers = num_layers
        self.target_vocab_size = input_vocab_size
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model) 
        # note: the embedding can be used by BERT and use look_up function
    
        self.pos_encoding = postional_encoder(maximum_position_encoding, self.d_model)
    
        self.encoder = [EncoderLayer(d_model, num_heads, dff, dropout_pro) for _ in range(num_layers)]
    
        self.dropout = tf.keras.layers.Dropout(dropout_pro)
    
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        
        # Embedding
        x = self.embedding(x) 
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        # positional Encoding
        x += self.pos_encoding[:, :seq_len, :]
        
        # Dropout
        x = self.dropout(x, training = training)
        
        # Encoder
        for i in range(self.num_layers):
            x = self.encoder[i](x, training, mask)
            
        return x

class Decoder(tf.keras.Model):
    '''
    1. Output Embedding 
    2. Positional Encoding 
    3. N decoder layers
    '''
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding=512, dropout_pro=0.1):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
    
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = postional_encoder(maximum_position_encoding, d_model)
    
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_pro)
                       for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_pro)
        # prediction layer
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}
    
        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
    
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)
      
        attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
        attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        x = self.final_layer(x)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    '''
    This optimizer is self-adpative learning rate
    lr = d_model^(-0.5) * min(step_num^(-0.5), step_num*warmup_steps**(-1.5))
    '''
    def __init__(self, d_model, warmup_steps = 4000):
        super(CustomSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        
        self.warmup_steps = warmup_steps
        
    def __call__(self, steps):
        arg1 = tf.math.rsqrt(steps)
        arg2 = steps * (self.warmup_steps**-1.5)
        
        return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1, arg2)


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
def loss_function(real, pred):
    '''
    This is loss function, we use cross 
    '''
    mask = tf.math.logical_not(tf.math.equal(real, 0)) #True or False ->0 or 1
    loss_ = loss_object(real, pred)
    
    mask = tf.cast(mask, dtype = tf.float32) #use for padding influence
    loss_ *= mask
    
    return tf.reduce_mean(loss_)

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, seq_len, d_model//num_heads)

def create_look_ahead_mask(size):
    #[0, 1, 1, 1, ..., 1], 
    #[0, 0, 1, 1, ..., 1]
    #...
    #[0, 0, 0, 0, ..,  1]
    mask = 1 - tf.linalg.band_part(tf.ones([size, size]), -1, 0)
    return mask  # (seq_len, seq_len)

def create_masks(inp, tar):

    enc_padding_mask = create_padding_mask(inp)
    dec_padding_mask = create_padding_mask(inp)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
  
    return enc_padding_mask, combined_mask, dec_padding_mask



  

   
    
    
    
    
    
    
    
    
