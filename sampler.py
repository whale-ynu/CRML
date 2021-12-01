import numpy
from queue import Queue
from scipy.sparse import lil_matrix


class RatingSampler(object):

    def __init__(self, user_item_matrix, batch_size, n_negative, check_negative=True):
        """
    
        :param user_item_matrix: the user-item matrix for positive user-item pairs
        :param batch_size: number of samples to return
        :param n_negative: number of negative samples per user-positive-item pair
        :param result_queue: the output queue
        :return: None
        """
        self.result_queue = Queue()
                
        self.user_item_matrix = lil_matrix(user_item_matrix)
        self.user_item_pairs = numpy.asarray(self.user_item_matrix.nonzero()).T
        self.user_to_positive_set = {u: set(row) for u, row in enumerate(self.user_item_matrix.rows)}
        self.batch_size = batch_size
        self.n_negative = n_negative
        self.check_negative = check_negative
    
    def generate_batches(self, triple_format=False): 
        numpy.random.shuffle(self.user_item_pairs)
        for i in range(int(len(self.user_item_pairs) / self.batch_size) + 1):
            user_positive_items_pairs = None
            if (i + 1) * self.batch_size < len(self.user_item_pairs):
                user_positive_items_pairs = self.user_item_pairs[i * self.batch_size: (i + 1) * self.batch_size, :]
            else:  # for the last mini_batch where the size is less than self.batch_size
                user_positive_items_pairs = self.user_item_pairs[i * self.batch_size:, :]
            # print(len(user_positive_items_pairs))
            # print(user_positive_items_pairs)
            # sample negative samples
            # negative_samples = numpy.random.randint(0, self.user_item_matrix.shape[1], size=(self.batch_size, self.n_negative))
            negative_samples = numpy.random.randint(0, self.user_item_matrix.shape[1], size=(len(user_positive_items_pairs), self.n_negative))
            # Check if we sample any positive items as negative samples.
            # Note: this step can be optional as the chance that we sample a positive item is fairly low given a
            # large item set.
            if self.check_negative:
                for user_positive, negatives, i in zip(user_positive_items_pairs, negative_samples, range(len(negative_samples))):
                    user = user_positive[0]
                    for j, neg in enumerate(negatives):
                        while neg in self.user_to_positive_set[user]: 
                            negative_samples[i, j] = neg = numpy.random.randint(0, self.user_item_matrix.shape[1])
            
            if triple_format is False:
                self.result_queue.put((user_positive_items_pairs, negative_samples, [idx for idx in range(len(user_positive_items_pairs))])) 
            elif triple_format is True:
                users, items, labels = [], [], []
                for i in range(len(user_positive_items_pairs)):
                    users.append(user_positive_items_pairs[i][0])
                    items.append(user_positive_items_pairs[i][1])
                    labels.append(self.user_item_matrix[user_positive_items_pairs[i][0], user_positive_items_pairs[i][1]])
                    for j in negative_samples[i]:
                        users.append(user_positive_items_pairs[i][0])
                        items.append(j)
                        labels.append(0.0)
                self.result_queue.put((users, items, labels)) 
        
    def is_empty(self):
        return self.result_queue.empty()
    
    def next_batch(self):
        return self.result_queue.get()


class TrustSampler(object):

    def __init__(self, trust_matrix):
        self.pairs, self.value = [], []
        self.cooc_matrix = lil_matrix(trust_matrix)
        self.row_to_col_set = {u: set(row) for u, row in enumerate(self.cooc_matrix.rows)}
        # make the trust relationship bidirectional
        for u in self.row_to_col_set:
            for i in self.row_to_col_set[u]:
                self.cooc_matrix[i, u] = 1
        
        self.row_to_col_set = {u: set(row) for u, row in enumerate(self.cooc_matrix.rows)}
        
    def next_batch(self, users_in_minibatch):
        trust_pairs_in_minibatch = []
        for u in users_in_minibatch:
            for i in self.row_to_col_set[u]:
                if i > u: trust_pairs_in_minibatch.append([u, i])
        return trust_pairs_in_minibatch
                
        
class PairSampler(object):

    def __init__(self, batch_size, input_source):
        self.result_queue = Queue()
        self.batch_size = batch_size
        self.rows = input_source['ROW']
        self.cols = input_source['COL']
        self.vals = input_source['VAL']
            
    def generate_batches(self): 
        rng_state = numpy.random.get_state()
        numpy.random.shuffle(self.rows)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(self.cols)
        numpy.random.set_state(rng_state)
        numpy.random.shuffle(self.vals)
        
        for i in range(int(len(self.vals) / self.batch_size) + 1):
            self.result_queue.put(i)       
            
    def is_empty(self):
        return self.result_queue.empty()
    
    def next_batch(self):
        i = self.result_queue.get()   
        if (i + 1) * self.batch_size < len(self.vals):
            batch_row = self.rows[i * self.batch_size: (i + 1) * self.batch_size]
            batch_col = self.cols[i * self.batch_size: (i + 1) * self.batch_size]
            batch_val = [[self.vals[k]] for k in range(i * self.batch_size, (i + 1) * self.batch_size)]
        else:
            batch_row = self.rows[i * self.batch_size:]
            batch_col = self.cols[i * self.batch_size:]
            batch_val = [[self.vals[k]] for k in range(i * self.batch_size, len(self.vals))]
        # print(len(batch_val))
        return list(zip(batch_row, batch_col)), batch_val
