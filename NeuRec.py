import sys 
sys.path.append("..") 
import tensorflow.compat.v1 as tf
from time import time
import numpy as np
import toolz
from evaluator import Evaluator
from sampler import RatingSampler
from datatool import DataProcessor
from scipy.sparse import vstack

  
class UNeuRec:
    '''
    Shuai Zhang, Lina Yao, Aixin Sun, Sen Wang, Guodong Long, Manqing Dong: NeuRec: On Nonlinear Transformation for Personalized Ranking. IJCAI 2018: 3669-3675
    '''

    def __init__(self, n_users, n_items, n_negatives, train,
                 reg_rate=0.001, learning_rate=0.001, reg=[0, 0], random_seed=2018):
        self.n_users = n_users
        self.n_items = n_items
        self.n_negatives = n_negatives
        self.train_csr = train.tocsr()
        self.learning_rate = learning_rate        
        self.random_seed = random_seed
        self.reg_rate = reg_rate
        self._init_graph()
        
    def extract_userfactor(self, user):
        train_rows = list()
        for u in user:
            train_rows.append(self.train_csr.getrow(u)) 
        sub_train = vstack(train_rows)
        coo = sub_train.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return (indices, coo.data, coo.shape)
  
    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.user_factor = tf.sparse_placeholder(tf.float32)
            self.item = tf.placeholder(dtype=tf.int32, shape=[None], name='item')
            self.label = tf.placeholder("float", [None], 'label')
            # self.isTrain = tf.placeholder(tf.bool, shape=())
    
            self.score_user_factor = tf.sparse_placeholder(tf.float32)
            self.maxk = tf.placeholder(tf.int32, shape=())
            # Variables.
            hidden_dim_1 = 150
            hidden_dim_2 = 150
            hidden_dim_3 = 150
            hidden_dim_4 = 150
            hidden_dim_5 = 40
            
            W1 = tf.Variable(tf.random_normal([self.n_items, hidden_dim_1]))
            W2 = tf.Variable(tf.random_normal([hidden_dim_1, hidden_dim_2]))
            W3 = tf.Variable(tf.random_normal([hidden_dim_2, hidden_dim_3]))
            W4 = tf.Variable(tf.random_normal([hidden_dim_3, hidden_dim_4]))
            W5 = tf.Variable(tf.random_normal([hidden_dim_4, hidden_dim_5]))
    
            b1 = tf.Variable(tf.random_normal([hidden_dim_1]))
            b2 = tf.Variable(tf.random_normal([hidden_dim_2]))
            b3 = tf.Variable(tf.random_normal([hidden_dim_3]))
            b4 = tf.Variable(tf.random_normal([hidden_dim_4]))
            b5 = tf.Variable(tf.random_normal([hidden_dim_5]))
            
            P = tf.Variable(tf.random_normal([self.n_items, hidden_dim_5], stddev=0.005))
                        # embeddings
            item_factor = tf.nn.embedding_lookup(P, self.item)
            # multiply perception layers

            layer_1 = tf.sigmoid(tf.sparse_tensor_dense_matmul (self.user_factor, W1) + b1)
            layer_2 = tf.sigmoid(tf.matmul(layer_1, W2) + b2)
            layer_3 = tf.sigmoid(tf.matmul(layer_2, W3) + b3)
            layer_4 = tf.sigmoid(tf.matmul(layer_3, W4) + b4)
            layer_5 = tf.sigmoid(tf.matmul(layer_4, W5) + b5)
            # prediction layer
            self.pred_y = tf.reduce_sum(tf.multiply(item_factor, layer_5), 1)
            
            self.loss = tf.reduce_sum(tf.square(self.label - self.pred_y)) + self.reg_rate * (
                tf.nn.l2_loss(W1) + tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3) + tf.nn.l2_loss(W4) + tf.nn.l2_loss(
                    W5) + tf.nn.l2_loss(P))
        
            # optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
            '''
            for evaluation part
            '''
            test_user_emb = tf.sigmoid(tf.sparse_tensor_dense_matmul(self.score_user_factor, W1) + b1)
            test_user_emb = tf.sigmoid(tf.matmul(test_user_emb, W2) + b2)
            test_user_emb = tf.sigmoid(tf.matmul(test_user_emb, W3) + b3)
            test_user_emb = tf.sigmoid(tf.matmul(test_user_emb, W4) + b4)
            test_user_emb = tf.sigmoid(tf.matmul(test_user_emb, W5) + b5)

            self.item_scores = tf.matmul(test_user_emb, P, transpose_b=True)
            
            self.top_k = tf.nn.top_k(self.item_scores, self.maxk)
            
            # initialization
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            self.graph.finalize()

    def train(self, log, sampler, train, test, iters, top_k):
        log.write("############################################### \n")
        test_users = list(set(test.nonzero()[0]))
        eval = Evaluator(train, test)
        for i in range(iters):
            print("Iter %d..." % (i + 1))
            log.write("Iter %d...\t" % (i + 1))

            t1 = time()             
            print("Optimizing  loss...") 
            loss = 0
            sampler.generate_batches(triple_format=True)
            
            while not sampler.is_empty():
                user, item, label = sampler.next_batch()
                
                _, batch_loss = self.sess.run((self.optimizer, self.loss), feed_dict={self.user_factor: self.extract_userfactor(user), self.item: item,
                                                                                      self.label:label})
                loss += batch_loss
                print('.', end='')
                
            t2 = time()
            
            print("\nTraining loss: %f  in %.2fs" % (loss, t2 - t1))
            log.write("Training loss: %f  in %.2fs \n" % (loss, t2 - t1))
            
            if (i + 1) % 1 == 0:
                test_recall = list()
                test_precision = list()
                test_ndcg = list()
                test_hitratio = list()
                # compute metrics in chunks to utilize speedup provided by Tensorflow
                for user_chunk in toolz.partition_all(50, test_users):
                     # compute the top (K +  Max Number Of Training Items for any user) items for each user
                    _, ranked_items_for_users = model.sess.run(model.top_k,
                                feed_dict={model.score_user_factor: self.extract_userfactor(user_chunk),
                                 model.maxk:top_k[-1] + eval.max_train_count})
                    
                    _r, _p, _n, _h = eval.evalRankPerformance(user_chunk, ranked_items_for_users, top_k)
                    
                    test_recall.extend(_r)
                    test_precision.extend(_p)
                    test_ndcg.extend(_n)
                    test_hitratio.extend(_h)

                for j in range(len(top_k)):
                    recall = 0
                    for m in range(len(test_recall)):
                        recall += test_recall[m][j]
                    precision = 0
                    for m in range(len(test_precision)):
                        precision += test_precision[m][j]
                    ndcg = 0
                    for m in range(len(test_ndcg)):
                        ndcg += test_ndcg[m][j]
                    hitratio = 0
                    for m in range(len(test_hitratio)):
                        hitratio += test_hitratio[m][j]
                    print("Top-%d Recall: %f Precision: %f NDCG: %f HR:%f" % (top_k[j], recall / len(test_recall),
                                                                                precision / len(test_precision),
                                                                                ndcg / len(test_ndcg),
                                                                                hitratio / len(test_hitratio)))
                    log.write("Top-%d Recall: %f Precision: %f NDCG: %f HR:%f\n" % (top_k[j], recall / len(test_recall),
                                                                                    precision / len(test_precision),
                                                                                    ndcg / len(test_ndcg),
                                                                                    hitratio / len(test_hitratio)))
                t3 = time()
                print("Eval costs: %f s\n" % (t3 - t2))
                log.write("Eval costs: %f s\n" % (t3 - t2))

            log.flush()


if __name__ == '__main__':
    BATCH_SIZE = [64, 64, 64]
    TOP_K = [1, 5, 10]
    MAX_ITERS = [30, 30, 20]
    N_NEGATIVE = [20, 20, 20]
    DATASET_NAMES = ['filmtrust', 'andriod_app', 'kindle-store']
    
        # get user-item matrix
    for i in range(len(DATASET_NAMES)):
        # get user-item matrix
        dp = DataProcessor('../datasets/' + DATASET_NAMES[i] + '/ratings.dat')
        n_users, n_items = dp.n_users_items()
        train, test = dp.split_ratings_by_leaveoneout()
        sampler = RatingSampler(train, batch_size=BATCH_SIZE[i], n_negative=N_NEGATIVE[i], check_negative=True)   
                            
        log = open('../log/ranking/' + DATASET_NAMES[i] + '.UNeuRec.log', 'a') 
        log.write("############################################### \n")
        log.write("n_negative=%d  \n" % (N_NEGATIVE[i]))
        log.write("batch_size=%d \n" % (BATCH_SIZE[i]))
        log.flush() 
        for n_neg in [10, 20, 40]:    
            model = UNeuRec(n_users=n_users, n_items=n_items, n_negatives=n_neg, train=train, learning_rate=0.001)
            model.train(log, sampler, train, test, MAX_ITERS[i], TOP_K)
        log.close()
