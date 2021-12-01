import sys 
sys.path.append("..") 
import tensorflow.compat.v1 as tf
from time import time
import toolz
from evaluator import Evaluator
from sampler import RatingSampler
from datatool import DataProcessor
from keras import backend as K

                        
class NeuMF:
    '''
    He X, Liao L, Zhang H, et al. Neural collaborative filtering, WWW2017: 173-182.
    '''

    def __init__(self, n_users, n_items, n_negatives, layers=[64, 32, 16, 8], reg_layers=[0.0, 0.0, 0.0, 0.0], reg_mf=0, master_learning_rate=0.001, reg=[0, 0], random_seed=2018):
        self.n_users = n_users
        self.n_items = n_items
        self.n_negatives = n_negatives
        self.reg_layers = reg_layers
        self.reg_mf = reg_mf
        self.layers = layers
        self.master_learning_rate = master_learning_rate        
        self.random_seed = random_seed
        self._init_graph()
            
    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
#            self.items = tf.placeholder(tf.int32, [None])
#            self.labels = tf.placeholder(tf.float32, [None])
            self.user_positive_items_pairs = tf.placeholder(tf.int64, [None, 2])
            self.negative_samples = tf.placeholder(tf.int32, [None, self.n_negatives])
            self.score_user_ids = tf.placeholder(tf.int32, [None])
            self.score_n_users = tf.placeholder(tf.int32, shape=())
            self.maxk = tf.placeholder(tf.int32, shape=())
            # Variables.
            self.weights = { 
                'mf_user_embeddings':tf.Variable(tf.random_normal([self.n_users, self.layers[-1]], mean=0.0, stddev=0.01, dtype=tf.float32)),
                'mf_item_embeddings':tf.Variable(tf.random_normal([self.n_items, self.layers[-1]], mean=0.0, stddev=0.01, dtype=tf.float32)),
                'mlp_user_embeddings':tf.Variable(tf.random_normal([self.n_users, int(self.layers[0] / 2)], mean=0.0, stddev=0.01, dtype=tf.float32)),
                'mlp_item_embeddings':tf.Variable(tf.random_normal([self.n_items, int(self.layers[0] / 2)], mean=0.0, stddev=0.01, dtype=tf.float32))
                }
            
            with tf.name_scope("mlp"):  
                self.mlp = []
                for idx in range(1, len(self.layers)):
                    self.mlp.append(tf.layers.Dense(units=self.layers[idx], kernel_regularizer=tf.keras.regularizers.l2(self.reg_layers[idx]), activation='relu', name='layer%d' % idx))
                    
            self.predict = tf.layers.Dense(units=1, activation='sigmoid', kernel_initializer='lecun_uniform')
            
            # for MLP
            mlp_user_latent = tf.nn.embedding_lookup(self.weights['mlp_user_embeddings'], self.user_positive_items_pairs[:, 0])
            mlp_item_latent = tf.nn.embedding_lookup(self.weights['mlp_item_embeddings'], self.user_positive_items_pairs[:, 1])
            
            self.mlp_vector = tf.concat([mlp_user_latent, mlp_item_latent], 1)
            
            for i in range(len(self.mlp)):
                 self.mlp_vector = self.mlp[i].apply(self.mlp_vector)  # (?,8)

            neg_item_embs = tf.nn.embedding_lookup(self.weights['mlp_item_embeddings'], self.negative_samples)  # (?,20,32)
            neg_vectors = tf.concat([tf.tile(tf.expand_dims(mlp_user_latent, 1), [1, self.n_negatives, 1]),  # (?,20,64)
                                     neg_item_embs], 2)
    
            for i in range(len(self.mlp)):
                 neg_vectors = self.mlp[i].apply(neg_vectors)  # (?,20,8)

            # for GMF 
            mf_user_latent = tf.nn.embedding_lookup(self.weights['mf_user_embeddings'], self.user_positive_items_pairs[:, 0])
            mf_item_latent = tf.nn.embedding_lookup(self.weights['mf_item_embeddings'], self.user_positive_items_pairs[:, 1])
            
            self.mf_vector = tf.multiply(mf_user_latent, mf_item_latent)
            
            self.predict_vector = tf.concat([self.mf_vector, self.mlp_vector], 1)  # (?,16)
            self.preditions = self.predict.apply(self.predict_vector)  # (?,1)
            
            neg_item_embs = tf.nn.embedding_lookup(self.weights['mf_item_embeddings'], self.negative_samples)
            neg_predictions = tf.multiply (tf.expand_dims(mf_user_latent, 1), neg_item_embs)  # (?,20,8)

            neg_predictions = tf.concat([neg_predictions, neg_vectors ], 2)  # (?,20,16)
            neg_predictions = self.predict.apply(neg_predictions)
			
             # Final prediction layer

            self.loss = -tf.reduce_sum(tf.log(self.preditions)) * self.n_negatives
            self.loss += -tf.reduce_sum(tf.log(1 - neg_predictions))\
            + tf.keras.regularizers.l2(0.001)(self.weights["mlp_user_embeddings"])\
            + tf.keras.regularizers.l2(0.001)(self.weights["mlp_item_embeddings"])\
            + tf.keras.regularizers.l2(0.001)(self.weights["mf_user_embeddings"])\
            + tf.keras.regularizers.l2(0.001)(self.weights["mf_item_embeddings"])
            
            self.optimizer = tf.train.AdamOptimizer(self.master_learning_rate).minimize(self.loss)
        
            '''
            for evaluation part
            '''
            mf_test_user_emb = tf.expand_dims(tf.nn.embedding_lookup(self.weights['mf_user_embeddings'], self.score_user_ids), 1)
            mf_test_item_emb = tf.expand_dims(self.weights['mf_item_embeddings'], 0)
            mf_test_vector = tf.multiply(mf_test_user_emb, mf_test_item_emb)
            
            mlp_test_user_emb = tf.expand_dims(tf.nn.embedding_lookup(self.weights['mlp_user_embeddings'], self.score_user_ids), 1)
            mlp_test_item_emb = tf.expand_dims(self.weights['mlp_item_embeddings'], 0)
            mlp_test_vector = tf.concat([tf.tile(mlp_test_user_emb, [1, self.n_items, 1]), tf.tile(mlp_test_item_emb, [self.score_n_users, 1, 1])], 2)
            
            for i in range(len(self.mlp)):
                 mlp_test_vector = self.mlp[i].apply(mlp_test_vector)
            
            test_vector = tf.concat([mf_test_vector, mlp_test_vector], 2)
            
            self.item_scores = tf.squeeze(self.predict.apply(test_vector))
            
            self.top_k = tf.nn.top_k(self.item_scores, self.maxk)
            
            # initialization
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            K.set_session(self.sess)
            self.sess.run(init)
            self.graph.finalize()

    def train(self, log, sampler, train, test, iters, top_k):
        
        log.write("embed_dim=%d \n" % (int(self.layers[0] / 2)))
        log.write("############################################### \n")

        test_users = list(set(test.nonzero()[0]))
        eval = Evaluator(train, test)
        for i in range(iters):
            print("Iter %d..." % (i + 1))
            log.write("Iter %d...\t" % (i + 1))

            t1 = time()             
            print("Optimizing  loss...") 
            loss = 0
            sampler.generate_batches()
            while not sampler.is_empty():
                ui_pos, ui_neg, _ = sampler.next_batch()

                _, batch_loss = self.sess.run((self.optimizer, self.loss), {self.user_positive_items_pairs: ui_pos,
                                                                      self.negative_samples: ui_neg})
                loss += batch_loss
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
                                                  feed_dict={model.score_user_ids: user_chunk,
                                                             model.score_n_users: len(user_chunk),
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
                    print("Top-%d Recall: %f Precision: %f NDCG: %f HR:%f\n" % (top_k[j], recall / len(test_recall),
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
    BATCH_SIZE = [64, 64, 64, 64]
    TOP_K = [1, 5, 10]
    MAX_ITERS = [100, 20, 20, 20]
    N_NEGATIVE = [20, 20, 20, 20]
    DATASET_NAMES = ['filmtrust', 'kindle_store']
    
        # get user-item matrix
    for i in range(len(DATASET_NAMES)):
        # get user-item matrix
        dp = DataProcessor('../datasets/' + DATASET_NAMES[i] + '/ratings.dat')
        n_users, n_items = dp.n_users_items()
        train, test = dp.split_ratings_by_leaveoneout()
        sampler = RatingSampler(train, batch_size=BATCH_SIZE[i], n_negative=N_NEGATIVE[i], check_negative=True)   
        
        log = open('../log/' + DATASET_NAMES[i] + '.NeuMF.log', 'a') 
        log.write("############################################### \n")
        log.write("n_negative=%d  \n" % (N_NEGATIVE[i]))
        log.write("batch_size=%d \n" % (BATCH_SIZE[i]))
        log.flush()      
        model = NeuMF(n_users=n_users, n_items=n_items, n_negatives=N_NEGATIVE[i], layers=[64, 32, 16, 8], reg_layers=[0.0, 0.0, 0.0, 0.0], master_learning_rate=0.0001)
        model.train(log, sampler, train, test, MAX_ITERS[i], TOP_K)
        log.close()

