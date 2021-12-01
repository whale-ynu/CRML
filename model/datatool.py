import csv 
import numpy as np
import random
from scipy.sparse import dok_matrix, lil_matrix, csr_matrix
from tqdm import tqdm


class DataProcessor:
    
    def __init__(self, rating_datafile, trust_datafile=None, binary=True):
        self.n_users = 0
        self.n_items = 0
        self.max_rating=0.0
        self.loadRatings(rating_datafile, binary)
        self.trust_matrix = None
        if trust_datafile is not None:
            self.loadTrusts(trust_datafile)
        
    def getRatingMatrix(self):
        return self.user_item_matrix
    
    def getTrustMatrix(self):
        return self.trust_matrix
    
    def n_users_items(self):
        return self.n_users, self.n_items

    
    def loadRatings(self, filepath, binary=True):
        # for storing original indices of users/ items
        self.userset, self.itemset = {}, {}  
        # for re-assigning indices for users/ items
        user_idx, item_idx = 0, 0 
        with open(filepath, 'r', encoding='utf-8') as csvin:
            reader = csv.reader(csvin, delimiter=' ')
            
            for line in reader:
                # print(line)
                u = line[0]
                if u not in self.userset:
                    self.userset[u] = user_idx
                    user_idx = user_idx + 1
                        
                i = line[1]
                if i not in self.itemset:
                    self.itemset[i] = item_idx
                    item_idx = item_idx + 1
                    
        self.n_users = len(self.userset)
        self.n_items = len(self.itemset)
        self.user_item_matrix = dok_matrix((self.n_users, self.n_items), dtype=np.float32) 
        with open(filepath, 'r', encoding='utf-8') as csvin:
            reader = csv.reader(csvin, delimiter=' ')
            for line in reader:
                u, i = line[0], line[1]
                if binary is True:
                    self.user_item_matrix[self.userset[u], self.itemset[i]] = 1.0
                else:
                    r =np.float(line[2])
                    self.user_item_matrix[self.userset[u], self.itemset[i]] = r
                    if r > self.max_rating:
                        self.max_rating = r
                
        print('#users: %d #items: %d #ratings: %d' % (self.n_users, self.n_items, len(self.user_item_matrix.nonzero()[0])))
        if binary is False:
            print('#max ratings: %f ' % (self.max_rating))
        
    def loadTrusts(self, filepath, binary=True):
        trustor, trustee = set(), set()
        self.trust_matrix = dok_matrix((self.n_users, self.n_users), dtype=np.float32)         
        with open(filepath, 'r', encoding='utf-8') as csvin:
            reader = csv.reader(csvin, delimiter=' ')
            for line in reader:
                u, f = line[0], line[1]
                if u in self.userset and f in self.userset:
                    # if(user_cooccur_matrix[userset[u],userset[f]]>0):
                        trustor.add(u)
                        trustee.add(f)
                        if binary is True:
                            self.trust_matrix[self.userset[u], self.userset[f]] = 1
                        else:
                            self.trust_matrix[self.userset[u], self.userset[f]] = np.float32(line[2])
        print('#trustors: %d #trustees: %d #trusts: %d' % (len(trustor), len(trustee), len(self.trust_matrix.nonzero()[0])))     
       
        #=======================================================================
        # num_trusts = []
        # for i, row_i in enumerate(lil_matrix(self.trust_matrix).rows):
        #     if len(row_i) > 0: num_trusts.append(len(row_i))
        # print("Averge trusts: %f Max: %f Min: %f" % (np.mean(num_trusts), np.max(num_trusts), np.min(num_trusts)))  
        #=======================================================================
    
    def split_ratings_by_leaveoneout(self, seed=2018):
        random.seed(seed)
        train = dok_matrix(self.user_item_matrix.shape)
        # validation = dok_matrix(user_item_matrix.shape)
        test = dok_matrix(self.user_item_matrix.shape)
        # convert it to lil format for fast row access
        new_matrix = lil_matrix(self.user_item_matrix)
        for user in tqdm(range(new_matrix.shape[0]), desc="Split data into train/test"):
            items = list(new_matrix.rows[user])
            selected = random.sample(items, 1)
            
            for i in set(items).difference(selected):
                train[user, i] = new_matrix[user, i]
            for i in selected:
                test[user, i] = new_matrix[user, i]
                
        print("{}/{} train/test samples".format(len(train.nonzero()[0]), len(test.nonzero()[0])))
        return train, test

    def split_ratings_by_ratio(self, split_ratio=(8, 2), seed=2018):
        # set the seed to have deterministic results
        np.random.seed(seed)
        train = dok_matrix(self.user_item_matrix.shape)
        # validation = dok_matrix(user_item_matrix.shape)
        test = dok_matrix(self.user_item_matrix.shape)
        # convert it to lil format for fast row access
        new_matrix = lil_matrix(self.user_item_matrix)
        for user in tqdm(range(new_matrix.shape[0]), desc="Split data into train/test"):
            items = list(new_matrix.rows[user])
            if len(items) >= 5:
                
                np.random.shuffle(items)
                
                train_count = int(len(items) * split_ratio[0] / sum(split_ratio))
                # valid_count = int(len(items) * split_ratio[1] / sum(split_ratio))
    
                for i in items[0: train_count]:
                    train[user, i] = new_matrix[user, i]
                for i in items[train_count: ]:
                    test[user, i] = new_matrix[user, i]
                
        print("{}/{} train/test samples".format(len(train.nonzero()[0]), len(test.nonzero()[0])))
        return train, test

    def save_dok_matrix(self, matrix, target_file):
        print("save to " + target_file)
        f = open(target_file, "w")
        row_indices = matrix.nonzero()[0]
        col_indices = matrix.nonzero()[1]
        for pos in range(len(row_indices)):
            f.write("%d\t%d\t%.0f\n" % (row_indices[pos], col_indices[pos], matrix[row_indices[pos], col_indices[pos]]))
        f.close()
        
    #===========================================================================
    # def save_and_sample_negative(self, test_matrix, n_negatives, target_file):
    #     user_to_positive_set = {u: set(row) for u, row in enumerate(lil_matrix(self.user_item_matrix).rows)}
    #     test_pairs = np.asarray(test_matrix.nonzero()).T
    #     # sample negative samples
    #     negative_samples = np.random.randint(0, test_matrix.shape[1], size=(test_matrix.shape[0], n_negatives))
    # 
    #     for test_pair, negatives, i in zip(test_pairs, negative_samples, range(len(negative_samples))):
    #         user = test_pair[0]
    #         for j, neg in enumerate(negatives):
    #             while neg in user_to_positive_set[user]: 
    #                 negative_samples[i, j] = neg = np.random.randint(0, test_matrix.shape[1])
    #     print("save to " + target_file)
    #     f = open(target_file, "w")
    #     
    #     for test_pair, negatives in zip(test_pairs, negative_samples):
    #         f.write("(%d,%d)" % (test_pair[0], test_pair[1]))
    #         for i in negatives: f.write("\t%d" % i)
    #         f.write('\n')
    #     f.close()
    #===========================================================================
     
    def est_cooccurrence_of_users(self, train_matrix, min_count=1):
        user_to_item_set = {u: set(row) for u, row in enumerate(lil_matrix(train_matrix).rows)}
        rows, cols, vals = list(), list(), list()
        count2 = 0
        count1 = 0
        for i in tqdm(range(train_matrix.shape[0]), desc="est_cooccurrence_of_users"):
            for j in range(i + 1, train_matrix.shape[0]):
                coo = len(user_to_item_set[i] & user_to_item_set[j])
                if coo >= min_count: 
                    rows.append(i)
                    cols.append(j)
                    vals.append(float(coo))
                if coo == 2:
                    count2 += 1
                if coo == 1:
                    count1 += 1
        print(count1)
        print(count2)
        print('max: %d, min:%d, total:%d' % (max(vals), min(vals), len(vals)))
        return {'ROW':rows, 'COL':cols, 'VAL':vals}
    
    def est_cooccurrence_of_items(self, train_matrix, min_count=1):
        item_to_user_set = {i: set(row) for i, row in enumerate(lil_matrix(train_matrix.transpose()).rows)}
        rows, cols, vals = list(), list(), list()
        count2 = 0
        count1 = 0
        for i in tqdm(range(train_matrix.shape[1]), desc="est_cooccurrence_of_items"):
            for j in range(i + 1, train_matrix.shape[1]):
                coo = len(item_to_user_set[i] & item_to_user_set[j])
                if coo >= min_count: 
                    rows.append(i)
                    cols.append(j)
                    vals.append(float(coo))
                if coo == 2:
                    count2 += 1
                if coo == 1:
                    count1 += 1
        print(count1)
        print(count2)
        print('max: %d, min:%d, total:%d' % (max(vals), min(vals), len(vals)))        
        return {'ROW':rows, 'COL':cols, 'VAL':vals}

    def est_sppmi_of_items(self, train_matrix, k_ns=1, min_count=1):
        item_to_user_set = {i: set(row) for i, row in enumerate(lil_matrix(train_matrix.transpose()).rows)}
        rows, cols, vals = list(), list(), list()
        for i in tqdm(range(train_matrix.shape[1]), desc="est_cooccurrence_of_items"):
            for j in range(i + 1, train_matrix.shape[1]):
                coo = len(item_to_user_set[i] & item_to_user_set[j])
                if coo >= min_count: 
                    rows.append(i)
                    cols.append(j)
                    vals.append(float(coo))
                    rows.append(j)
                    cols.append(i)
                    vals.append(float(coo))
        print('max: %d, min:%d, total:%d' % (max(vals), min(vals), len(vals)))     
        X = csr_matrix((vals, (rows, cols)), shape=(train_matrix.shape[1], train_matrix.shape[1]))
        count = np.asarray(X.sum(axis=1)).ravel()
        n_pairs = X.data.sum()
        
        M = X.copy()
        for i in tqdm(range(train_matrix.shape[1]), desc="est_sppmi_of_items"):
            # lo, hi, d, idx = self.get_row(M, i)
            lo, hi = M.indptr[i], M.indptr[i + 1]
            d, idx = M.data[lo:hi], M.indices[lo:hi]
            M.data[lo:hi] = np.log(d * n_pairs / (count[i] * count[idx]))
        M.data[M.data < 0] = 0
        M.eliminate_zeros()
        # number of negative samples
        M_ns = M.copy()
        if k_ns > 1:
            offset = np.log(k_ns)
        else:
            offset = 0.
        M_ns.data -= offset
        M_ns.data[M_ns.data < 0] = 0
        M_ns.eliminate_zeros()
        return M_ns

        
class Similarity:

    def __init__(self, train_matrix, trust_matrix):
        self.train_matrix = train_matrix
        self.trust_matrix = trust_matrix
        
    def est_rating_similarity(self, social_sim):
        similarity = dok_matrix((self.train_matrix.shape[0], self.train_matrix.shape[0]), dtype=np.float32)
        user_to_item_set = {u: set(row) for u, row in enumerate(lil_matrix(self.train_matrix).rows)}
        
        for i, j in zip(social_sim.nonzero()[0], social_sim.nonzero()[1]):
            # Jaccard similarity between user i and user j
            similarity[i, j] = 0
            part1 = len(user_to_item_set[i] & user_to_item_set[j])
            if part1 > 0:
                part2 = len(user_to_item_set[i] | user_to_item_set[j])
                if part2 > 0:
                    similarity[i, j] = float(part1) / part2
            similarity[i, j] = 0.5 * similarity[i, j] + 0.5 * social_sim[i, j]
        # print(len(similarity.nonzero()[0]))
        return similarity        
    
    def est_social_similarity(self):
        similarity = dok_matrix(self.trust_matrix.shape, dtype=np.float32)
        lil_trust_matrix = lil_matrix(self.trust_matrix)
        tranlil_trust_matrix = lil_matrix(self.trust_matrix.transpose())
        for i, row_i in enumerate(lil_trust_matrix.rows):
            for j in set(row_i):
                row_j = tranlil_trust_matrix.getrow(j)
                similarity[i, j] = float(row_j.getnnz()) / float(len(row_i) + row_j.getnnz())
                # print(similarity[i,j])
        # print(len(similarity.nonzero()[0]))
        return similarity
    
    def est_similarity(self):
        social_sim = self.est_social_similarity()
        return self.est_rating_similarity(social_sim)
    
                 

    
from time import time
    
if __name__ == '__main__':
    
    dp = DataProcessor('../datasets/ml-1m/ratings.dat', binary=False)
    # dp.save_dok_matrix(train, 'D:/workspace/librec-v2.0/data/' + dataset_name + '/given_testset/train/train.dat')
    # dp.save_dok_matrix(test, 'D:/workspace/librec-v2.0/data/' + dataset_name + '/given_testset/test//test.dat')
    # dp.save_dok_matrix(trust_matrix, 'D:/workspace/librec-v2.0/data/' + dataset_name + '/trust/trusts.dat')
    # sim=Similarity(train, dataprocessor.trust_matrix)
    # similarity = sim.est_similarity()
    # train, test=dp.split_ratings_by_ratio()
    #===========================================================================
    train, test = dp.split_ratings_by_leaveoneout()
    coos = dp.est_cooccurrence_of_users(train)
    coos = dp.est_cooccurrence_of_items(train)
    # dp.est_sppmi_of_items(train)
    # dp.save_and_sample_negative(test, 100, 'datasets/lastfm/lastfm.test.negative')
    # dp.save_dok_matrix(train, 'datasets/lastfm/lastfm.train.rating')
    # dp.save_dok_matrix(test, 'datasets/lastfm/lastfm.test.rating')
    #===========================================================================
    
    dp = DataProcessor('datasets/filmtrust/ratings.txt')
    # dp.save_dok_matrix(train, 'D:/workspace/librec-v2.0/data/' + dataset_name + '/given_testset/train/train.dat')
    # dp.save_dok_matrix(test, 'D:/workspace/librec-v2.0/data/' + dataset_name + '/given_testset/test//test.dat')
    # dp.save_dok_matrix(trust_matrix, 'D:/workspace/librec-v2.0/data/' + dataset_name + '/trust/trusts.dat')
    # sim=Similarity(train, dataprocessor.trust_matrix)
    # similarity = sim.est_similarity()
    # train, test=dp.split_ratings_by_ratio()
   
    #===========================================================================
    # 
    # dp.save_and_sample_negative(test, 100, 'datasets/filmtrust/filmtrust.test.negative')
    # dp.save_dok_matrix(train, 'datasets/filmtrust/filmtrust.train.rating')
    # dp.save_dok_matrix(test, 'datasets/filmtrust/filmtrust.test.rating')
    #===========================================================================
    #===========================================================================
    # cc = CooccurrenceCounter(train, 3)
    # t1 = time()
    # cc.est_cooccurrence_of_users()
    # cc.est_cooccurrence_of_items()
    # t2 = time()
    # print(t2 - t1)
    # 
    #===========================================================================
