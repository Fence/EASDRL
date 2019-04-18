#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
# Project:  Extracting Action Sequences Based on Deep Reinforcement Learning
# Module:   Clustering
# Author:   Wenfeng Feng 
# Time:     2019.03
################################################################################

import os
import ipdb
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import load_pkl, save_pkl, get_time


def prepare_tfidf_data():
    """
    Compute tfidf features for texts
    """
    filename = 'data/online_test/tf_idf_data_max_feature20000.pkl'
    if os.path.exists(filename):
        print('Loading data from %s' % filename)
        return load_pkl(filename)
    else:
        texts = load_pkl('data/home_and_garden_500_words_with_title.pkl')
        print('Vectorizing texts ...')
        converter = TfidfVectorizer(decode_error='ignore', 
                                    stop_words='english', 
                                    max_features=20000)
        tf_idf = converter.fit_transform([' '.join(t['sent']) for t in texts])
        print('n_samples: %d  n_features: %d' % tf_idf.shape)
        save_pkl(tf_idf, filename)
        return tf_idf


def clustering():
    """
    Clustering all unlabeled texts
    """
    data = prepare_tfidf_data()
    SSE = []
    SilhouetteScores = []
    Labels = []
    with open('data/online_test/clustering_result_%s.txt' % get_time(), 'w') as f:
        for k in range(2, 21):
            print('\nClustering, n_cluster = %d' % k)
            f.write('\nClustering, n_cluster = %d\n' % k)
            model = KMeans(n_clusters=k, init='k-means++', n_jobs=-1)
            labels = model.fit(data).labels_
            
            print(' model is trained, trying to compute silhouette score')
            score = metrics.silhouette_score(data, labels)
            Labels.append(labels)
            SilhouetteScores.append(score)
            SSE.append(model.inertia_)
            print('SSE: {}\n'.format(SSE))
            print('SilhouetteScores: {}\n'.format(SilhouetteScores))
            f.write(' SSE: {}\n SilhouetteScores: {}\n'.format(SSE, SilhouetteScores))
    
    all_records = {'SSE': SSE, 'scores': SilhouetteScores, 'labels': Labels}
    save_pkl(all_records, 'data/online_test/clustering_records.pkl')
    # x = np.arange(1, 11)
    # plt.plot(x, SSE, 'o-')
    # plt.xlabel('k')
    # plt.ylabel('SSE')
    # plt.show()
    # plt.savefig('data/kmeans_clustering.pdf', format='pdf')
    print('Success!')


def text_classification():
    """
    Classify all unlabeled texts according to the clustering results 
    """
    filename = 'data/online_test/label2text.pkl'
    if os.path.exists(filename):
        label2text = load_pkl(filename)
    else:
        records = load_pkl('data/online_test/clustering_records.pkl')
        scores = records['scores']
        best_i = scores.index(max(scores[:10])) # maximum categories = 11
        labels = records['labels'][best_i]
        best_k = best_i + 2 # k in [2, 20], i in [0, 18]
        label2text = {c: [] for c in range(best_k)}
        for ind, label in enumerate(labels):
            label2text[label].append(ind)
        save_pkl(label2text, filename)

    for label, text_inds in label2text.items():
        print('label: {}  n_samples: {}'.format(label, len(text_inds)))
    return label2text



if __name__ == '__main__':
    # clustering()
    text_classification()