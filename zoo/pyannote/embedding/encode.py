import glob
import os
from pprint import pprint

import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist
from tqdm import tqdm

from soundutils.speaker import speaker_embedding
from test.testings import get_testfile

np.set_printoptions(precision=2, suppress=True)

if __name__ == '__main__':
    files = []
    names = []
    # dataset_dir = get_testfile('assets', 'speakers')
    dataset_dir = '/data/arknights_jp_nested'
    for name in os.listdir(dataset_dir):
        new_files = glob.glob(os.path.join(dataset_dir, name, '*.wav'))
        files.extend(new_files)
        names.extend([name] * len(new_files))
    pprint(files)

    embeddings = []
    for file in tqdm(files):
        embeddings.append(speaker_embedding(file))
    embs = np.stack(embeddings)
    distance = cdist(embs, embs, metric="cosine")
    # print(distance)

    names = np.array(names)
    g = names == names[..., None]
    # print(g)

    idx = np.arange(names.shape[0])
    msk = idx != idx[..., None]
    # print(g & msk)

    x = distance[g & msk]
    y = distance[~g]


    # plt.figure(figsize=(10, 6))
    # plt.boxplot([x, y], labels=['Group X', 'Group Y'])
    # plt.title('Distribution of Group X and Group Y')
    # plt.ylabel('Value')
    # plt.show()

    # TODO: determine a threshold between x and y

    def find_optimal_threshold(positive_scores, negative_scores, max_samples: int = 100000):
        if positive_scores.shape[0] > max_samples:
            positive_scores = np.random.choice(positive_scores, size=max_samples, replace=False)
        if negative_scores.shape[0] > max_samples:
            negative_scores = np.random.choice(negative_scores, size=max_samples, replace=False)
        # 合并并排序所有分数
        all_scores = np.concatenate([positive_scores, negative_scores])
        all_scores.sort()

        # 计算正样本和负样本的数量
        n_pos = len(positive_scores)
        n_neg = len(negative_scores)

        # 初始化变量
        tp = n_pos
        fp = n_neg
        best_f1 = 0
        best_threshold = None

        # 使用numpy的searchsorted来快速找到每个阈值对应的TP和FP
        pos_ranks = np.searchsorted(all_scores, positive_scores, side='right')
        neg_ranks = np.searchsorted(all_scores, negative_scores, side='right')

        scs = all_scores

        # 计算每个可能的阈值的F1分数
        scores, f1s, ps, rs = [], [], [], []
        for i, threshold in enumerate(tqdm(scs)):
            tp -= np.sum(pos_ranks == i)
            fp -= np.sum(neg_ranks == i)

            precision = tp / (tp + fp) if tp + fp > 0 else 0
            recall = tp / n_pos if n_pos > 0 else 0

            f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
            scores.append(threshold)
            ps.append(precision)
            rs.append(recall)
            f1s.append(f1)

        # plt.plot(scores, f1s, label='F1')
        # plt.plot(scores, ps, label='Percentage')
        # plt.plot(scores, rs, label='Recall')
        # plt.legend()
        # plt.show()

        return best_threshold, best_f1


    optimal_threshold, max_f1 = find_optimal_threshold(y, x, max_samples=100000)
    print(f"Optimal threshold: {optimal_threshold}")
    print(f"Max F1 score: {max_f1}")
