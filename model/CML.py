import sys 
sys.path.append("..") 
import numpy
import tensorflow.compat.v1 as tf
from time import time
import toolz
from evaluator import Evaluator
from sampler import RatingSampler
from datatool import DataProcessor


class CML:
    '''
    Hsieh C K, Yang L, Cui Y, et al. Collaborative metric learning[C], WWW2017: 193-201.
    '''

    def __init__(self, n_users, n_items, embed_dim=50, margin=1.5, layers=[100, 100], master_learning_rate=0.001,
                 clip_norm=1.0, use_rank_weight=True, use_cov_loss=False, cov_loss_weight=10, random_seed=2018):
        """
        :param n_users: number of users
        :param n_items: number of items
        :param embed_dim: embedding size
        :param margin: margin threshold for hinge loss
        :param master_learning_rate: master learning rate
        :param clip_norm: clip norm threshold (default 1.0)
        """

        self.n_users = n_users
        self.n_items = n_items
        self.embed_dim = embed_dim

        self.clip_norm = clip_norm
        self.margin = margin
        self.layers = layers
        
        self.master_learning_rate = master_learning_rate
        self.use_rank_weight = use_rank_weight
        self.use_cov_loss = use_cov_loss
        self.cov_loss_weight = cov_loss_weight

        self.random_seed = random_seed
        self._init_graph()

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing: input data, variables, model, rating_loss, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():  # , tf.device('/cpu:0'):
            # Set graph level random seed
            tf.set_random_seed(self.random_seed)
            # Input data.
            self.user_positive_items_pairs = tf.placeholder(tf.int64, [None, 2])
            self.negative_samples = tf.placeholder(tf.int32, [None, None])
            self.score_user_ids = tf.placeholder(tf.int32, [None])
            self.row_indices = tf.placeholder(tf.int32, [None])
            self.maxk = tf.placeholder(tf.int32, shape=())
            # Variables.
            self.weights = self._initialize_weights()
            
            '''
            Subgraph 1
            '''
            # user embedding (N, K)
            users = tf.nn.embedding_lookup(self.weights['user_embeddings'], self.user_positive_items_pairs[:, 0], name="users")
            # positive item embedding (N, K)
            pos_items = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.user_positive_items_pairs[:, 1], name="pos_items")
            
            # positive item to user distance (N)
            pos_distances = tf.reduce_sum(tf.squared_difference(users, pos_items), 1, name="pos_distances")
    
            # negative item embedding (N, K, W)
            neg_items = tf.nn.embedding_lookup(self.weights['item_embeddings'], self.negative_samples, name="neg_items")
            # neg_items=self.item_projection(neg_items)
            neg_items = tf.transpose(neg_items, (0, 2, 1))
            # distance to negative items (N x W)
            distance_to_neg_items = tf.reduce_sum(tf.squared_difference(tf.expand_dims(users, -1), neg_items), 1, name="distance_to_neg_items")
    
            # best negative item (among W negative samples) their distance to the user embedding (N)
            closest_negative_item_distances = tf.reduce_min(distance_to_neg_items, 1, name="closest_negative_distances")
    
            # compute hinge rating_loss (N)
            loss_per_pair = tf.maximum(pos_distances - closest_negative_item_distances + self.margin, 0, name="pair_loss")
    
            if self.use_rank_weight:
                # indicator matrix for impostors (N x W)
                impostors = (tf.expand_dims(pos_distances, -1) - distance_to_neg_items + self.margin) > 0
                # approximate the rank of positive item by (number of impostor / W per user-positive pair)
                rank = tf.reduce_mean(tf.cast(impostors, dtype=tf.float32), 1, name="rank_weight") * self.n_items
                # apply rank weight
                loss_per_pair *= tf.log(rank + 1)
    
            # the embedding trust_loss
            self.rating_loss = tf.reduce_sum(loss_per_pair, name="rating_loss")

            if self.use_cov_loss:
                self.rating_loss += self.covariance_loss()
                
            self.optimizer1 = tf.train.AdamOptimizer(self.master_learning_rate).minimize(self.rating_loss)
        
            with tf.control_dependencies([self.optimizer1]):
                tf.assign(self.weights['user_embeddings'], tf.clip_by_norm(self.weights['user_embeddings'], self.clip_norm, axes=[1]))
                tf.assign(self.weights['item_embeddings'], tf.clip_by_norm(self.weights['item_embeddings'], self.clip_norm, axes=[1]))
            
            '''
            for evaluation part
            '''
            # (N_USER_IDS, 1, K)
            test_users = tf.expand_dims(tf.nn.embedding_lookup(self.weights['user_embeddings'], self.score_user_ids), 1)
            # (1, N_ITEM, K)
            test_items = tf.expand_dims(self.weights['item_embeddings'], 0)
            # score = minus distance (N_USER, N_ITEM)
            self.item_scores = -tf.reduce_sum(tf.squared_difference(test_users, test_items), 2)
            
            self.top_k = tf.nn.top_k(self.item_scores, self.maxk)
            
            # initialization
            init = tf.global_variables_initializer()
            self.sess = tf.Session()
            self.sess.run(init)
            self.graph.finalize()

    def _initialize_weights(self):
        all_weights = dict()
        # _________ Embedding Layer _____________
        all_weights['user_embeddings'] = tf.Variable(tf.random_normal([self.n_users, self.embed_dim], stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
        all_weights['item_embeddings'] = tf.Variable(tf.random_normal([self.n_items, self.embed_dim], stddev=1 / (self.embed_dim ** 0.5), dtype=tf.float32))
        return all_weights
   
    def covariance_loss(self):
        X = tf.concat((self.weights['item_embeddings'], self.weights['user_embeddings']), 0)
        n_rows = tf.cast(tf.shape(X)[0], tf.float32)
        X = X - (tf.reduce_mean(X, axis=0))
        cov = tf.matmul(X, X, transpose_a=True) / n_rows
        return tf.reduce_sum(tf.matrix_set_diag(cov, tf.zeros(self.embed_dim, tf.float32))) * self.cov_loss_weight

    def train(self, log, dataset_name, sampler, train, test, iters, top_k):
        """
        Optimize the self. TODO: implement early-stopping
        :param dataset_name: dataset used
        :param sampler: mini-batch sampler for rating data
        :param train: train user-item matrix
        :param test: test user-item matrix
        :param iters: max number of iterations
        :param top_k: top-k in performance metrics
        :return: None
        """
        log.write("embed_dim=%d \n" % (self.embed_dim))
        log.write("margin=%.2f \n" % (self.margin))
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
            # run n mini-batches
            t1 = time()
            
            print("Optimizing  rating_loss...") 
            sampler.generate_batches()
            while not sampler.is_empty():
                ui_pos, ui_neg, _ = sampler.next_batch()
                _, loss1 = self.sess.run((self.optimizer1, self.rating_loss), feed_dict=
                                         {self.user_positive_items_pairs: ui_pos,
                                          self.negative_samples: ui_neg})
    
                losses.append(loss1)
            t2 = time()
            print("\nTraining loss: %f in %.2fs" % (numpy.sum(losses), t2 - t1))
            log.write("Training loss: %f in %.2fs\n" % (numpy.sum(losses), t2 - t1))
            
            if (i + 1) % 5 == 0:
                
                test_recall = list()
                test_precision = list()
                test_ndcg = list()
                test_hitratio = list()
                # compute metrics in chunks to utilize speedup provided by Tensorflow
                for user_chunk in toolz.partition_all(100, test_users):
                    # compute the top (K +  Max Number Of Training Items for any user) items for each user
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
    BATCH_SIZE = [1024, 1024, 1024, 1024, 1024]
    TOP_K = [1, 5, 10]
    MAX_ITERS = [100, 200, 50, 200, 150]
    N_NEGATIVE = [20, 20, 20, 20, 20]
    DATASET_NAMES = ['filmtrust', 'lastfm', 'douban', 'ciaodvd', 'epinion']
    
        # get user-item matrix
    for i in range(len(DATASET_NAMES)):
        # get user-item matrix
        dp = DataProcessor('../datasets/' + DATASET_NAMES[i] + '/ratings.dat')
        n_users, n_items = dp.n_users_items()
        train, test = dp.split_ratings_by_leaveoneout()
        # create warp samplers
        sampler = RatingSampler(train, batch_size=BATCH_SIZE[i], n_negative=N_NEGATIVE[i], check_negative=True)   
        log = open('../log/' + DATASET_NAMES[i] + '.cml.log', 'a') 
        log.write("############################################### \n")
        log.write("n_negative=%d  \n" % (N_NEGATIVE[i]))
        log.write("batch_size=%.2f \n" % (BATCH_SIZE[i]))
        log.flush()      
#         for embed_dim in [50, 100, 200]:
#             for margin in [0.5, 1.0, 1.5]:
        for embed_dim in [100]:
            for margin in [2.0]:
                # Train a user-item joint embedding, where the items a user likes will be pulled closer to this users.
                # Once the embedding is trained, the recommendations are made by finding the k-Nearest-Neighbor to each user.
                model = CML(n_users, n_items, embed_dim=embed_dim, layers=[embed_dim, embed_dim], margin=margin, clip_norm=1, master_learning_rate=0.001)
                model.train(log, DATASET_NAMES[i], sampler, train, test, MAX_ITERS[i], TOP_K)
        log.close()
