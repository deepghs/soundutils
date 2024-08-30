import glob
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
    for name in ['texas', 'kroos', 'surtr', 'mlynar', 'nian']:
        new_files = glob.glob(get_testfile('assets', 'speakers', name, '*.wav'))
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

    plt.figure(figsize=(10, 6))
    plt.boxplot([x, y], labels=['Group X', 'Group Y'])
    plt.title('Distribution of Group X and Group Y')
    plt.ylabel('Value')
    plt.show()

    # TODO: determine a threshold between x and y
