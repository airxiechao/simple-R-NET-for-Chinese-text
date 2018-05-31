import tensorflow as tf
from tensorflow.contrib import rnn
import json
import numpy as np

class R_NET:
    
    def __init__(self, batch_size=10, p_length=300, q_length=30, emb_dim=300):
        self.batch_size = batch_size
        self.p_length = p_length
        self.q_length = q_length
        self.emb_dim = emb_dim
        self.state_size = 75
        self.span_length = 20
        
    def mat_weight_mul(self, mat, weight):
        # [batch_size, n, m] * [m, p] = [batch_size, n, p]
        mat_shape = mat.get_shape().as_list()
        weight_shape = weight.get_shape().as_list()
        assert(mat_shape[-1] == weight_shape[0])
        mat_reshape = tf.reshape(mat, [-1, mat_shape[-1]]) # [batch_size * n, m]
        mul = tf.matmul(mat_reshape, weight) # [batch_size * n, p]
        return tf.reshape(mul, [-1, mat_shape[1], weight_shape[-1]])
    
    def build_model(self):
        
        # placeholders
        paragraph = tf.placeholder(tf.float32, [self.batch_size, self.p_length, self.emb_dim])
        question = tf.placeholder(tf.float32, [self.batch_size, self.q_length, self.emb_dim])
        answer_si = tf.placeholder(tf.float32, [self.batch_size, self.p_length])
        answer_ei = tf.placeholder(tf.float32, [self.batch_size, self.p_length])
        
        # encoding
        unstack_question = tf.unstack(question, self.q_length, 1)
        unstack_paragraph = tf.unstack(paragraph, self.p_length, 1)
        with tf.variable_scope('encoding') as scope:
            fw_cell = rnn.BasicLSTMCell(self.state_size)
            bw_cell = rnn.BasicLSTMCell(self.state_size)
            
            q_enc, _, _ = rnn.static_bidirectional_rnn(fw_cell, bw_cell, unstack_question, dtype=tf.float32)
            tf.get_variable_scope().reuse_variables()
            p_enc, _, _ = rnn.static_bidirectional_rnn(fw_cell, bw_cell, unstack_paragraph, dtype=tf.float32)
            
            u_Q = tf.stack(q_enc, 1) # [batch_size, q_length, 2 * state_size]
            u_P = tf.stack(p_enc, 1) # [batch_size, p_length, 2 * state_size]
            
        # question-paragraph match
        v_P = []
        with tf.variable_scope('QP_match') as scope:
            W_uQ = tf.Variable(tf.truncated_normal([2*self.state_size, self.state_size]))
            W_uP = tf.Variable(tf.truncated_normal([2*self.state_size, self.state_size]))
            W_vP = tf.Variable(tf.truncated_normal([self.state_size, self.state_size]))
            W_g_QP = tf.Variable(tf.truncated_normal([4*self.state_size, 4*self.state_size]))
            B_v_QP = tf.Variable(tf.truncated_normal([self.state_size]))
            
            qp_match_cell = rnn.BasicLSTMCell(self.state_size)
            qp_match_state = qp_match_cell.zero_state(self.batch_size, dtype=tf.float32)
            
            for t in range(self.p_length):
                
                # c_t
                W_uQ_u_Q = self.mat_weight_mul(u_Q, W_uQ)
                u_tP = tf.concat( [tf.reshape(u_P[:, t, :], [self.batch_size, 1, -1])] * self.q_length, 1)
                W_uP_u_tP = self.mat_weight_mul(u_tP , W_uP)
                
                if t == 0:
                    tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP)
                else:
                    v_t1P = tf.concat( [tf.reshape(v_P[t-1], [self.batch_size, 1, -1])] * self.q_length, 1)
                    W_vP_v_t1P = self.mat_weight_mul(v_t1P, W_vP)
                    tanh = tf.tanh(W_uQ_u_Q + W_uP_u_tP + W_vP_v_t1P)
                    
                s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(B_v_QP, [-1, 1])))
                a_t = tf.nn.softmax(s_t, 1)
                tiled_a_t = tf.concat( [tf.reshape(a_t, [self.batch_size, -1, 1])] * 2 * self.state_size , 2)
                c_t = tf.reduce_sum( tf.multiply(tiled_a_t, u_Q) , 1) # [batch_size, 2 * state_size]
                
                # gate
                u_tP_c_t = tf.expand_dims( tf.concat( [tf.squeeze(u_P[:, t, :]), c_t], 1), 1)
                g_t = tf.sigmoid( self.mat_weight_mul(u_tP_c_t, W_g_QP) )
                u_tP_c_t_star = tf.squeeze(tf.multiply(u_tP_c_t, g_t))
                
                qp_match_output, qp_match_state = qp_match_cell(u_tP_c_t_star, qp_match_state)
                v_P.append(qp_match_output)
                
        v_P = tf.stack(v_P, 1) # [batch_size, p_length, state_size]
        
        # self-match
        SM_star = []
        with tf.variable_scope('Self_match') as scope:
            W_smP1 = tf.Variable(tf.truncated_normal([self.state_size, self.state_size]))
            W_smP2 = tf.Variable(tf.truncated_normal([self.state_size, self.state_size]))
            W_g_SM = tf.Variable(tf.truncated_normal([2*self.state_size, 2*self.state_size]))
            B_v_SM = tf.Variable(tf.truncated_normal([self.state_size]))
            
            for t in range(self.p_length):
                
                # s_t
                W_p1_v_P = self.mat_weight_mul(v_P, W_smP1) # [batch_size, p_length, state_size]
                tiled_v_tP = tf.concat( [tf.reshape(v_P[:, t, :], [self.batch_size, 1, -1])] * self.p_length, 1)
                W_p2_v_tP = self.mat_weight_mul(tiled_v_tP , W_smP2)
                tanh = tf.tanh(W_p1_v_P + W_p2_v_tP)
            
                s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(B_v_SM, [-1, 1])))
                a_t = tf.nn.softmax(s_t, 1)
                tiled_a_t = tf.concat( [tf.reshape(a_t, [self.batch_size, -1, 1])] * self.state_size , 2)
                c_t = tf.reduce_sum( tf.multiply(tiled_a_t, v_P) , 1)
            
                # gate
                v_tP_c_t = tf.expand_dims( tf.concat( [tf.squeeze(v_P[:, t, :]), c_t], 1), 1)
                g_t = tf.sigmoid( self.mat_weight_mul(v_tP_c_t, W_g_SM) )
                v_tP_c_t_star = tf.squeeze(tf.multiply(v_tP_c_t, g_t))
                
                SM_star.append(v_tP_c_t_star)
            
            SM_star = tf.stack(SM_star, 1)
            unstacked_SM_star = tf.unstack(SM_star, self.p_length, 1)
            
            SM_fw_cell = rnn.BasicLSTMCell(self.state_size)
            SM_bw_cell = rnn.BasicLSTMCell(self.state_size)
            
            SM_outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(SM_fw_cell, SM_bw_cell, unstacked_SM_star, dtype=tf.float32)
            h_P = tf.stack(SM_outputs, 1) # [batch_size, p_length, 2 * state_size]
        
        # output
        p = [None for _ in range(2)]
        with tf.variable_scope('Ans_ptr') as scope:
            W_ruQ = tf.Variable(tf.truncated_normal([2*self.state_size, 2*self.state_size]))
            W_vQ = tf.Variable(tf.truncated_normal([self.state_size, 2*self.state_size]))
            W_VrQ = tf.Variable(tf.truncated_normal([self.q_length, self.state_size]))
            B_v_rQ = tf.Variable(tf.truncated_normal([2*self.state_size]))
            
            # r_Q
            W_ruQ_u_Q = self.mat_weight_mul(u_Q, W_ruQ) # [batch_size, q_length, 2 * state_size]
            W_vQ_V_rQ = tf.matmul(W_VrQ, W_vQ)
            W_vQ_V_rQ = tf.stack([W_vQ_V_rQ]*self.batch_size, 0) # stack -> [batch_size, q_length, 2 * state_size]
            tanh = tf.tanh(W_ruQ_u_Q + W_vQ_V_rQ)
            s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(B_v_rQ, [-1, 1])))
            a_t = tf.nn.softmax(s_t, 1)
            tiled_a_t = tf.concat( [tf.reshape(a_t, [self.batch_size, -1, 1])] * 2 * self.state_size , 2)
            r_Q = tf.reduce_sum( tf.multiply(tiled_a_t, u_Q) , 1) # [batch_size, 2 * state_size]
        
            # answer pointer
            W_hP = tf.Variable(tf.truncated_normal([2*self.state_size, self.state_size]))
            W_ha = tf.Variable(tf.truncated_normal([2*self.state_size, self.state_size]))
            B_v_ap = tf.Variable(tf.truncated_normal([self.state_size]))
            
            h_a = None
            
            ans_ptr_cell = rnn.BasicLSTMCell(2*self.state_size)
            ans_ptr_cell_state = ans_ptr_cell.zero_state(self.batch_size, dtype=tf.float32)
            for t in range(2):
                W_hP_h_P = self.mat_weight_mul(h_P, W_hP)
                
                if t == 0:
                    h_t1a = r_Q
                else:
                    h_t1a = h_a
        
                tiled_h_t1a = tf.concat( [tf.reshape(h_t1a, [self.batch_size, 1, -1])] * self.p_length, 1)
                W_ha_h_t1a = self.mat_weight_mul(tiled_h_t1a , W_ha)
                tanh = tf.tanh(W_hP_h_P + W_ha_h_t1a)
                s_t = tf.squeeze(self.mat_weight_mul(tanh, tf.reshape(B_v_ap, [-1, 1])))
                a_t = tf.nn.softmax(s_t, 1)
                tiled_a_t = tf.concat( [tf.reshape(a_t, [self.batch_size, -1, 1])] * 2 * self.state_size , 2)
                c_t = tf.reduce_sum( tf.multiply(tiled_a_t, h_P) , 1) # [batch_size, 2 * state_size]
            
                p[t] = a_t
                
                if t == 0:
                    h_a, ans_ptr_cell_state = ans_ptr_cell(c_t, (ans_ptr_cell_state.c, r_Q) )
                else:
                    pass
                    
        # loss
        p1 = p[0]
        p2 = p[1]
        
        answer_si_idx = tf.cast(tf.argmax(answer_si, 1), tf.int32)
        answer_ei_idx = tf.cast(tf.argmax(answer_ei, 1), tf.int32)
        
        batch_idx = tf.reshape(tf.range(0, self.batch_size), [-1,1])
        answer_si_re = tf.reshape(answer_si_idx, [-1,1])
        batch_idx_si = tf.concat([batch_idx, answer_si_re],1)
        answer_ei_re = tf.reshape(answer_ei_idx, [-1,1])
        batch_idx_ei = tf.concat([batch_idx, answer_ei_re],1)
    
        log_prob = tf.multiply(tf.gather_nd(p1, batch_idx_si), tf.gather_nd(p2, batch_idx_ei))
        loss = -tf.reduce_sum(tf.log(log_prob+0.0000001))
        
        # accuracy
        prob = []
        search_range = self.p_length - self.span_length
        for i in range(search_range):
            for j in range(self.span_length):
                prob.append(tf.multiply(p1[:, i], p2[:, i+j]))
        prob = tf.stack(prob, axis = 1)
        argmax_idx = tf.argmax(prob, axis=1)
        
        pred_si = argmax_idx / self.span_length
        pred_ei = pred_si + tf.cast(tf.mod(argmax_idx , self.span_length), tf.float64)
        
        correct = tf.logical_and(tf.equal(tf.cast(pred_si, tf.int64), tf.cast(answer_si_idx, tf.int64)), 
                                 tf.equal(tf.cast(pred_ei, tf.int64), tf.cast(answer_ei_idx, tf.int64)))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        
        input_tensors = {
            'p': paragraph,
            'q': question,
            'a_si':answer_si,
            'a_ei':answer_ei,
        }
        
        return input_tensors, loss, accuracy, pred_si, pred_ei
        
        
class DataProcessor:
    def __init__(self, batch_size=10, p_length=300, q_length=30, emb_dim=300):
        self.batch_size = batch_size
        self.p_length = p_length
        self.q_length = q_length
        self.emb_dim = emb_dim
        
        self.data = self.load_data('paragraph.json')
        self.vec = self.load_vec('wiki.zh.vec')
        
    def load_data(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
        
    def load_vec(self, path):
        vec = {}
        with open(path, 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            n, d = map(int, f.readline().split())
            for line in f:
                tokens = line.rstrip().split(' ')
                vec[tokens[0]] = list(map(float, tokens[1:]))
        return vec
        
    def word2vec(self, word):
        if word in self.vec:
            return self.vec[word]
        else:
            return np.zeros((self.emb_dim))
        
    def get_emb(self, text, length):
        emb = np.zeros((length, self.emb_dim))
        for i in range(min(length, len(text))):
            emb[i] = self.word2vec(text[i])
            
        return emb
        
    def get_num_samples(self):
        num = 0
        for d in self.data:
            num += len(d['qas'])
            
        return num
        
    def gen_test(self):
        while True:
            paragraph = np.zeros((2, self.p_length, self.emb_dim))
            question = np.zeros((2, self.q_length, self.emb_dim))
            answer_si = np.zeros((2, self.p_length))
            answer_ei = np.zeros((2, self.p_length))
                            
            p = input('paragraph:')
            q = input('question:')
            
            paragraph[0] = self.get_emb(p, self.p_length)
            paragraph[1] = self.get_emb(p, self.p_length)
            question[0] = self.get_emb(q, self.q_length)
            question[1] = self.get_emb(q, self.q_length)
            
            yield {
                'paragraph_text': p,
                'question_text': q,
                'paragraph': paragraph,
                'question': question,
                'answer_si': answer_si,
                'answer_ei': answer_ei
            }
        
    def gen_training_batch(self):
        while True:
            count = 0
            for d in self.data:
                p = d['paragraph']
                for qa in d['qas']:
                    if count == 0:
                        paragraph = np.zeros((self.batch_size, self.p_length, self.emb_dim))
                        question = np.zeros((self.batch_size, self.q_length, self.emb_dim))
                        answer_si = np.zeros((self.batch_size, self.p_length))
                        answer_ei = np.zeros((self.batch_size, self.p_length))
            
                    q = qa['question']
                    si = qa['start']
                    ei = qa['end']
                    
                    if si >= self.p_length or ei >= self.p_length:
                        continue
                    
                    paragraph[count] = self.get_emb(p, self.p_length)
                    question[count] = self.get_emb(q, self.q_length)
                    answer_si[count][si] = 1.0
                    answer_ei[count][ei] = 1.0
                    
                    count += 1
                    
                    if count % self.batch_size == 0:
                        yield {
                            'paragraph_text': p,
                            'question_text': q,
                            'si': si,
                            'ei': ei,
                            'paragraph': paragraph,
                            'question': question,
                            'answer_si': answer_si,
                            'answer_ei': answer_ei
                        }
                        
                        count = 0
            
        
def run(start_epoch=0):
    batch_size = 10
    p_length = 300
    q_length = 30
    emb_dim = 300
    
    print('load data...')
    dp = DataProcessor(batch_size, p_length, q_length, emb_dim)
    
    num_samples = dp.get_num_samples()
    num_batches = int(num_samples / batch_size)
    num_epochs = 200
    
    print('build model...')
    rnet_model = R_NET(batch_size, p_length, q_length, emb_dim)
    input_tensors, loss, acc, pred_si, pred_ei = rnet_model.build_model()
    train_op = tf.train.AdadeltaOptimizer(1.0, rho=0.95, epsilon=1e-06,).minimize(loss)
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    if start_epoch > 0:
        saver.restore(sess, "model/rnet_epoch_{}.ckpt".format(start_epoch-1))
    else:
        init = tf.global_variables_initializer()
        sess.run(init)
    
    print('training...')
    gen = dp.gen_training_batch()
    for ei in range(start_epoch, num_epochs):
        for bi in range(num_batches):
            tensor_dict = next(gen)
            feed_dict = {
                input_tensors['p']: tensor_dict['paragraph'],
                input_tensors['q']: tensor_dict['question'],
                input_tensors['a_si']: tensor_dict['answer_si'],
                input_tensors['a_ei']: tensor_dict['answer_ei'],
            }
            _, loss_value, accuracy, predictions_si, predictions_ei = sess.run(
                [train_op, loss, acc, pred_si, pred_ei], feed_dict=feed_dict)

            print("{} epoch {} batch, Loss:{:.2f}, Acc:{:.2f}".format(ei, bi, loss_value, accuracy))
        
        if ei % 50 == 0:
            saver.save(sess, "model/rnet_epoch_{}.ckpt".format(ei))

def test():
    p_length = 300
    q_length = 30
    emb_dim = 300
    
    print('load data...')
    dp = DataProcessor(2, p_length, q_length, emb_dim)
    
    print('build model...')
    rnet_model = R_NET(2, p_length, q_length, emb_dim)
    input_tensors, loss, acc, pred_si, pred_ei = rnet_model.build_model()
    
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.InteractiveSession(config=config)
    saver.restore(sess, "model/rnet_epoch_{}.ckpt".format(150))
    
    gen = dp.gen_test()
    while True:
        tensor_dict = next(gen)
        feed_dict = {
            input_tensors['p']: tensor_dict['paragraph'],
            input_tensors['q']: tensor_dict['question'],
            input_tensors['a_si']: tensor_dict['answer_si'],
            input_tensors['a_ei']: tensor_dict['answer_ei'],
        }
        predictions_si, predictions_ei = sess.run([pred_si, pred_ei], feed_dict=feed_dict)
            
        p = tensor_dict['paragraph_text']
        q = tensor_dict['question_text']
        
        print('paragraph: '+p)
        print('question: '+q)
        print('answer: '+p[int(predictions_si[0]):int(predictions_ei[0]+1)])

            
if __name__ == '__main__':
    run()
    test()
