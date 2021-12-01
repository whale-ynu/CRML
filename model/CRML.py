import sys 
sys.path.append("..") 
import numpy
import tensorflow.compat.v1 as tf
from time import time
import toolz
from evaluator import Evaluator
from sampler import RatingSampler, PairSampler
from datatool import DataProcessor
import pickle


class CRML:
    '''
    Hao Wu, Qimin Zhou, Rencan Nie, Jinde Cao, 
    Effective metric learning with co-occurrence embedding for collaborative recommendations, 
    Neural Networks, 2020,124:308-318. 
    https://doi.org/10.1016/j.neunet.2020.01.021.
    '''    
    def __init__(self, n_users, n_items, embed_dim=50, margin=1.5, scaling_factor=3 / 4, cooccurrence_cap=100, alpha=0, beta=0, master_learning_rate=0.001,
                 clip_norm=1.0, use_rank_weight=False, use_cov_loss=False, cov_loss_weight=10, random_seed=2018, mode='USER'):
        """
        :param n_users: number of users i.e. |U|
        :param n_items: number of items i.e. |V|
        :param embed_dim: embedding size i.e. K (default 20)
        :param features: (optional) the feature vectors of items, shape: (|V|, N_Features).
               Set it to None will disable feature trust_loss(default: None)
        :param margin: hinge trust_loss threshold i.e. z
        :param master_learning_rate: master learning rate for AdaGrad
        :param clip_norm: clip norm threshold (default 1.0)
        """
        self.mode = mode
        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.clip_norm = clip_norm
        self.margin = margin
        
        self.master_learning_rate = master_learning_rate
        self.use_rank_weight = use_rank_weight
        self.use_cov_loss = use_cov_loss
        self.cov_loss_weight = cov_loss_weight

        self.alpha = alpha
        self.beta = beta
        self.scaling_factor = scaling_factor
        self.cooccurrence_cap = cooccurrence_cap
        
        self.random_seed = random_seed
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, cml_loss, glove_optimizer1
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.user_positive_items_pairs = tf.placeholder(tf.int32, [None, 2])
            self.negative_samples = tf.placeholder(tf.int32, [None, None])
            self.score_user_ids = tf.placeholder(tf.int32, [None])
            self.score_n_users = tf.placeholder(tf.int32, shape=())
            self.row_indices = tf.placeholder(tf.int32, [None])
            self.maxk = tf.placeholder(tf.int32, shape=())
            # Variables.
            self.cml_weights = { 
                'user_embeddings':tf.Variable(tf.random_normal([self.n_users, self.embed_dim], stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32)),
                'item_embeddings':tf.Variable(tf.random_normal([self.n_items, self.embed_dim], stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
                }
            
            self.coemb_weights = {
                'co_user_embeddings':     tf.Variable(tf.random_normal([self.n_users, self.embed_dim], stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32)),
                'user_biases':         tf.Variable(tf.random_uniform([self.n_users], 1.0, -1.0)),
                'co_item_embeddings':     tf.Variable(tf.random_normal([self.n_items, self.embed_dim], stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32)),
                'item_biases':         tf.Variable(tf.random_uniform([self.n_items], 1.0, -1.0))
            }
    
            '''
            Subgraph 1
            '''
            # user embedding (N, K)
            users = tf.nn.embedding_lookup(self.cml_weights['user_embeddings'], self.user_positive_items_pairs[:, 0], name="users")
            # positive item embedding (N, K)
            pos_items = tf.nn.embedding_lookup(self.cml_weights['item_embeddings'], self.user_positive_items_pairs[:, 1], name="pos_items")
                    # positive item to user distance (N)
            pos_distances = tf.reduce_sum(tf.squared_difference(users, pos_items), 1, name="pos_distances")
    
            # negative item embedding (N, K, W)
            neg_items = tf.nn.embedding_lookup(self.cml_weights['item_embeddings'], self.negative_samples, name="neg_items")
            # neg_items=self.item_projection(neg_items)
            neg_items = tf.transpose(neg_items, (0, 2, 1))
            # distance to negative items (N x W)
            distance_to_neg_items = tf.reduce_sum(tf.squared_difference(tf.expand_dims(users, -1), neg_items), 1, name="distance_to_neg_items")
    
            # best negative item (among W negative samples) their distance to the user embedding (N)
            closest_negative_item_distances = tf.reduce_min(distance_to_neg_items, 1, name="closest_negative_distances")
            
            # compute hinge cml_loss (N)
            loss_per_pair = tf.maximum(pos_distances - closest_negative_item_distances + self.margin, 0, name="pair_loss")
    
            if self.use_rank_weight:
                # indicator matrix for impostors (N x W)
                impostors = (tf.expand_dims(pos_distances, -1) - distance_to_neg_items + self.margin) > 0
                # approximate the rank of positive item by (number of impostor / W per user-positive pair)
                rank = tf.reduce_mean(tf.cast(impostors, dtype=tf.float32), 1, name="rank_weight") * self.n_items
                # apply rank weight
                loss_per_pair *= tf.log(rank + 1)
    
            # the embedding cml_loss
            self.cml_loss = tf.reduce_sum(loss_per_pair, name="cml_loss")

            if self.use_cov_loss: 
                self.cml_loss += self.covariance_loss()
                
            # regluarization part
            if self.mode == 'USER' or self.mode == 'BOTH':
                self.all_users_cml = tf.unique(self.user_positive_items_pairs[:, 0]).y
                # construct a regularization term for users
                self.cml_loss += self.alpha * tf.reduce_sum(tf.squared_difference(tf.nn.embedding_lookup(self.cml_weights['user_embeddings'], self.all_users_cml),
                                                                           tf.nn.embedding_lookup(self.coemb_weights['co_user_embeddings'], self.all_users_cml)))
            if self.mode == 'ITEM' or self.mode == 'BOTH':
                # find the column indices of negative items with the closest distances
                column_indices_of_mindists = tf.cast(tf.argmin(distance_to_neg_items, axis=1), dtype=tf.int32)
                # construct the tensor indices of the best negative items
                tensor_indices_of_mindists = tf.concat([tf.expand_dims(self.row_indices, 1), tf.expand_dims(column_indices_of_mindists, 1)], axis=1)
                # gather the best negative items
                neg_items_with_mindists = tf.gather_nd(self.negative_samples, tensor_indices_of_mindists)
                # union with positive items
                self.all_items_cml = tf.unique(tf.concat([self.user_positive_items_pairs[:, 1], neg_items_with_mindists], axis=0)).y
                # construct a regularization term for items
                self.cml_loss += self.beta * tf.reduce_sum(tf.squared_difference(tf.nn.embedding_lookup(self.cml_weights['item_embeddings'], self.all_items_cml),
                                                                           tf.nn.embedding_lookup(self.coemb_weights['co_item_embeddings'], self.all_items_cml)))
            
            self.cml_optimizer = tf.train.AdamOptimizer(self.master_learning_rate).minimize(self.cml_loss)
        
            with tf.control_dependencies([self.cml_optimizer]):
                tf.assign(self.cml_weights['user_embeddings'], tf.clip_by_norm(self.cml_weights['user_embeddings'], self.clip_norm, axes=[1]))
                tf.assign(self.cml_weights['item_embeddings'], tf.clip_by_norm(self.cml_weights['item_embeddings'], self.clip_norm, axes=[1]))
                
            '''
            Subgraph 2
            '''
            # constants for Glove
            count_max = tf.constant([self.cooccurrence_cap], dtype=tf.float32)
            scaling_factor = tf.constant([self.scaling_factor], dtype=tf.float32)
            # placeholders
            self.user_pairs = tf.placeholder(tf.int32, shape=[None, 2])
            self.cooccurrence_count1 = tf.placeholder(tf.float32, shape=[None, 1])
            
            focal_user_embedding = tf.nn.embedding_lookup(self.coemb_weights['co_user_embeddings'], self.user_pairs[:, 0])
            context_user_embedding = tf.nn.embedding_lookup(self.coemb_weights['co_user_embeddings'], self.user_pairs[:, 1])
            focal_user_bias = tf.expand_dims(tf.nn.embedding_lookup(self.coemb_weights['user_biases'], self.user_pairs[:, 0]), 1)
            context_user_bias = tf.expand_dims(tf.nn.embedding_lookup(self.coemb_weights['user_biases'], self.user_pairs[:, 1]), 1)
                
            weighting_factor1 = tf.minimum(1.0, tf.pow(tf.div(self.cooccurrence_count1 + 1, count_max), scaling_factor))
            embedding_product1 = tf.reduce_sum(tf.multiply(focal_user_embedding, context_user_embedding), axis=1, keepdims=True)
            log_cooccurrences1 = tf.log(tf.to_float(self.cooccurrence_count1 + 1))
            distance_expr1 = tf.square(tf.add_n([embedding_product1, focal_user_bias, context_user_bias, tf.negative(log_cooccurrences1)]))
                
            # the embedding cml_loss
            self.glove_loss1 = tf.reduce_sum(tf.multiply(weighting_factor1, distance_expr1))
                
            self.glove_optimizer1 = tf.train.AdamOptimizer(self.master_learning_rate).minimize(self.glove_loss1)
            with tf.control_dependencies([self.glove_optimizer1]):
                 tf.assign(self.coemb_weights['co_user_embeddings'], tf.clip_by_norm(self.coemb_weights['co_user_embeddings'], self.clip_norm, axes=[1]))
            
            # placeholders
            self.item_pairs = tf.placeholder(tf.int32, shape=[None, 2])
            self.cooccurrence_count2 = tf.placeholder(tf.float32, shape=[None, 1])
            
            focal_item_emb = tf.nn.embedding_lookup(self.coemb_weights['co_item_embeddings'], self.item_pairs[:, 0])
            context_item_emb = tf.nn.embedding_lookup(self.coemb_weights['co_item_embeddings'], self.item_pairs[:, 1])
            focal_item_bias = tf.expand_dims(tf.nn.embedding_lookup(self.coemb_weights['item_biases'], self.item_pairs[:, 0]), 1)
            context_item_bias = tf.expand_dims(tf.nn.embedding_lookup(self.coemb_weights['item_biases'], self.item_pairs[:, 1]), 1)
                
            weighting_factor2 = tf.minimum(1.0, tf.pow(tf.div(self.cooccurrence_count2 + 1, count_max), scaling_factor))
            embedding_product2 = tf.reduce_sum(tf.multiply(focal_item_emb, context_item_emb), axis=1, keepdims=True)
            log_cooccurrences2 = tf.log(tf.to_float(self.cooccurrence_count2 + 1))
            distance_expr2 = tf.square(tf.add_n([embedding_product2, focal_item_bias, context_item_bias, tf.negative(log_cooccurrences2)]))
                
                # the embedding cml_loss
            self.glove_loss2 = tf.reduce_sum(tf.multiply(weighting_factor2, distance_expr2))
                
            self.glove_optimizer2 = tf.train.AdamOptimizer(self.master_learning_rate).minimize(self.glove_loss2)
            with tf.control_dependencies([self.glove_optimizer2]):
                 tf.assign(self.coemb_weights['co_item_embeddings'], tf.clip_by_norm(self.coemb_weights['co_item_embeddings'], self.clip_norm, axes=[1]))
            
            '''
            for evaluation part
            '''
            # (N_USER_IDS, 1, K)
            test_users = tf.expand_dims(tf.nn.embedding_lookup(self.cml_weights['user_embeddings'], self.score_user_ids), 1)
            # (1, N_ITEM, K)
            test_items = tf.expand_dims(self.cml_weights['item_embeddings'], 0)
            # score = minus distance (N_USER, N_ITEM)
            self.item_scores = -tf.reduce_sum(tf.squared_difference(test_users, test_items), 2)
            
            self.top_k = tf.nn.top_k(self.item_scores, self.maxk)
            
            # initialization
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            self.graph.finalize()
   
    def covariance_loss(self):
        X = tf.concat((self.cml_weights['item_embeddings'], self.cml_weights['user_embeddings']), 0)
        n_rows = tf.cast(tf.shape(X)[0], tf.float32)
        X = X - (tf.reduce_mean(X, axis=0))
        cov = tf.matmul(X, X, transpose_a=True) / n_rows
        return tf.reduce_sum(tf.matrix_set_diag(cov, tf.zeros(self.embed_dim, tf.float32))) * self.cov_loss_weight

    def train(self, log, cml_sampler, user_pair_sampler, item_pair_sampler, train, test, iters, top_k):
        """
        Optimize the self. TODO: implement early-stopping
        :param cml_sampler: mini-batch cml_sampler for rating data
        :param train: train user-item matrix
        :param test: test user-item matrix
        :param iters: max number of iterations
        :param top_k: top-k in performance metrics
        :return: None
        """
        log.write("embed_dim=%d \n" % (self.embed_dim))
        log.write("margin=%.2f \n" % (self.margin))
        log.write("alpha=%.4f \n" % (self.alpha))
        log.write("beta=%.4f \n" % (self.beta))
        log.write("cooccurrence_cap=%d \n" % (self.cooccurrence_cap))
        log.write("############################################### \n")

        # sample some users to calculate recall validation
        # test_users = numpy.random.choice(list(set(test.nonzero()[0])), size=1000, replace=True)
        test_users = list(set(test.nonzero()[0]))
        eval = Evaluator(train, test)
        for i in range(iters):
            print("Iter %d..." % (i + 1))
            log.write("Iter %d...\t" % (i + 1))
            
            # TODO: early stopping based on validation recall
            # train self.
            losses = []
            # run _n mini-batches
            t1 = time()
            if self.mode == 'USER' or self.mode == 'BOTH':
                print("Optimizing  coemb_loss1...") 
                coemb_loss1 = 0
                user_pair_sampler.generate_batches() 
                while not user_pair_sampler.is_empty():
                    user_pairs, cooccurrence_count = user_pair_sampler.next_batch()
                    _, loss1 = self.sess.run((self.glove_optimizer1, self.glove_loss1),
                                             {self.user_pairs:user_pairs, self.cooccurrence_count1:cooccurrence_count})
                    losses.append(loss1)
                    coemb_loss1 += loss1
            
            if self.mode == 'ITEM' or self.mode == 'BOTH':
                print("Optimizing  coemb_loss2...") 
                coemb_loss2 = 0
                item_pair_sampler.generate_batches() 
                while not item_pair_sampler.is_empty():
                    item_pairs, cooccurrence_count = item_pair_sampler.next_batch()
                    _, loss2 = self.sess.run((self.glove_optimizer2, self.glove_loss2),
                                             {self.item_pairs:item_pairs, self.cooccurrence_count2:cooccurrence_count})
                    losses.append(loss2)
                    coemb_loss2 += loss2
                
            print("Optimizing  cml_loss...") 
            cml_loss = 0
            cml_sampler.generate_batches()
            trust_pairs = []
            while not cml_sampler.is_empty():
                ui_pos, ui_neg, index = cml_sampler.next_batch()
                _, loss = self.sess.run((self.cml_optimizer, self.cml_loss), {self.user_positive_items_pairs: ui_pos,
                                                                                  self.negative_samples: ui_neg,
                                                                                  self.row_indices:index})
                losses.append(loss)
                cml_loss += loss
                
            t2 = time()
            
            if self.mode == 'ITEM':
                coemb_loss = coemb_loss2
            if self.mode == 'USER':
                coemb_loss = coemb_loss1
            if self.mode == 'ITEM' or self.mode == 'USER':
                print("\nTraining loss: %f (%f, %f) in %.2fs" % (numpy.sum(losses), coemb_loss, cml_loss, t2 - t1))
                log.write("Training loss: %f (%f, %f) in %.2fs \n" % (numpy.sum(losses), coemb_loss, cml_loss, t2 - t1))
            if self.mode == 'BOTH':
                print("\nTraining loss: %f (%f, %f, %f) in %.2fs" % (numpy.sum(losses), coemb_loss1, coemb_loss2, cml_loss, t2 - t1))
                log.write("Training loss: %f (%f, %f, %f) in %.2fs \n" % (numpy.sum(losses), coemb_loss1, coemb_loss2, cml_loss, t2 - t1))
            if self.mode == 'NONE':
                print("\nTraining loss: %f  in %.2fs" % (numpy.sum(losses), t2 - t1))
                log.write("Training loss: %f  in %.2fs \n" % (numpy.sum(losses), t2 - t1))
            
            if (i + 1) % 5 == 0:
                test_recall = list()
                test_precision = list()
                test_ndcg = list()
                test_hitratio = list()
                # compute metrics in chunks to utilize speedup provided by Tensorflow
                for user_chunk in toolz.partition_all(100, test_users):
                    _, ranked_items_for_users = model.sess.run(model.top_k,
                                                  feed_dict={model.score_user_ids: user_chunk,
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
                    print("Top-%d Recall: %f Precision: %f NDCG: %f HR: %f\n" % (top_k[j], recall / len(test_recall), precision / len(test_precision), ndcg / len(test_ndcg), hitratio / len(test_hitratio)))
                    log.write("Top-%d Recall: %f Precision: %f NDCG: %f HR: %f\n" % (top_k[j], recall / len(test_recall), precision / len(test_precision), ndcg / len(test_ndcg), hitratio / len(test_hitratio)))
                t3 = time()
                print("Eval costs: %f s\n" % (t3 - t2))
                log.write("Eval costs: %f s\n" % (t3 - t2))

            log.flush()


if __name__ == '__main__':
    BATCH_SIZE = [1024, 1024, 2048]
    TOP_K = [1, 5, 10]
    MAX_ITERS = [30, 100, 30]
    N_NEGATIVE = [20, 20, 20]
    DATASET_NAMES = ['andriod_app', 'ml-100k', 'filmtrust', 'lastfm', 'ml-100k']
    MODE = ['USER', 'ITEM', 'BOTH', 'NONE']
    
        # get user-item matrix
    for i in range(len(DATASET_NAMES)):
        # get user-item matrix
        dp = DataProcessor('../datasets/' + DATASET_NAMES[i] + '/ratings.dat')
        n_users, n_items = dp.n_users_items()
        train, test = dp.split_ratings_by_leaveoneout()
        pickle.dump(dp.est_cooccurrence_of_users(train, 2), open('datasets/' + DATASET_NAMES[i] + '/user_cooc.pkl', "wb"))
        pickle.dump(dp.est_cooccurrence_of_items(train), open('datasets/' + DATASET_NAMES[i] + '/item_cooc.pkl', "wb"))
        cml_sampler = RatingSampler(train, batch_size=BATCH_SIZE[i], n_negative=N_NEGATIVE[i], check_negative=True)
        user_cooc = pickle.load(open('datasets/' + DATASET_NAMES[i] + '/user_cooc.pkl', "rb"))   
        user_pair_sampler = PairSampler(BATCH_SIZE[i], user_cooc) 
        item_cooc = pickle.load(open('datasets/' + DATASET_NAMES[i] + '/item_cooc.pkl', "rb"))  
        item_pair_sampler = PairSampler(BATCH_SIZE[i], item_cooc) 
        
        log = open('../log/' + DATASET_NAMES[i] + '.CRML.log', 'a') 
        log.write("############################################### \n")
        log.write("n_negative=%d  \n" % (N_NEGATIVE[i]))
        log.write("batch_size=%d \n" % (BATCH_SIZE[i]))
        log.flush()      
        
        for embed_dim in [100]:
            for margin in [2.0]:
                
                #===============================================================
                # model = CRML(n_users=n_users, n_items=n_items, embed_dim=embed_dim, margin=margin, clip_norm=1, alpha=0, beta=0,  master_learning_rate=0.001, mode=MODE[3])
                # model.train(log, cml_sampler, user_pair_sampler, item_pair_sampler, train, test, MAX_ITERS[i], TOP_K)
                # 
                #===============================================================
                model = CRML(n_users=n_users, n_items=n_items, embed_dim=embed_dim, margin=margin, clip_norm=1, alpha=1.0, beta=0, master_learning_rate=0.001, mode=MODE[0])
                model.train(log, cml_sampler, user_pair_sampler, item_pair_sampler, train, test, MAX_ITERS[i], TOP_K)
                        
                model = CRML(n_users=n_users, n_items=n_items, embed_dim=embed_dim, margin=margin, clip_norm=1, alpha=0, beta=1.0, master_learning_rate=0.001, mode=MODE[1])
                model.train(log, cml_sampler, user_pair_sampler, item_pair_sampler, train, test, MAX_ITERS[i], TOP_K)
                
                model = CRML(n_users=n_users, n_items=n_items, embed_dim=embed_dim, margin=margin, clip_norm=1, alpha=1.0, beta=1.0, master_learning_rate=0.001, mode=MODE[2])
                model.train(log, cml_sampler, user_pair_sampler, item_pair_sampler, train, test, MAX_ITERS[i], TOP_K)
                        
                for alpha in [0, 0.01, 0.1, 1.0]:
                    for beta in [0, 0.01, 0.1, 1.0]:
                        
                        model = CRML(n_users=n_users, n_items=n_items, embed_dim=embed_dim, margin=margin, clip_norm=1, alpha=alpha, beta=beta, master_learning_rate=0.001, mode=MODE[2])
                        model.train(log, cml_sampler, user_pair_sampler, item_pair_sampler, train, test, MAX_ITERS[i], TOP_K)
                #===============================================================
                # for cooccurrence_cap in [10, 20, 40, 80, 160, 320]:
                #      model = CRML(n_users, n_items, embed_dim=embed_dim, margin=margin, clip_norm=1, alpha=1.0, cooccurrence_cap=cooccurrence_cap, master_learning_rate=0.001, mode=MODE[0])
                #      model.train(log, DATASET_NAMES[i], cml_sampler, user_pair_sampler, item_pair_sampler, train, test, MAX_ITERS[i], TOP_K)
                #===============================================================
        log.close()
