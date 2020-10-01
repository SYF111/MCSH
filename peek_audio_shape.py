import math
import pickle
import numpy as np
import h5py


def chunks(arr, m):
    n = int(math.ceil(len(arr) / float(m)))
    if n == 0:
        n = 1
    return [arr[i:i + n] for i in range(0, len(arr), n)]


def main():
    video_features_file = h5py.File('data/features/utterances_final/resnet_pool5.hdf5')
    segmented_video_feature = {}
    for idx, v in video_features_file.items():
        assert v.shape[1] == 2048
        a = chunks(v, 32)
        if len(a) >= 32:
            a = a[:32]
            b = np.array([np.mean(feature_vector, axis=0) for feature_vector in a])
            segmented_video_feature[idx] = b
        else:
            c = np.zeros((32, 2048))
            b = np.array([np.mean(feature_vector, axis=0) for feature_vector in a])
            c[:len(b), :] = b
            segmented_video_feature[idx] = c

    pickle.dump(segmented_video_feature,open('segmented_video_feature.p','wb'))


mean_video_feature = {}
video_features_file = h5py.File('data/features/utterances_final/resnet_pool5.hdf5')
segmented_video_feature = {}
for idx, v in video_features_file.items():
    assert v.shape[1] == 2048
    b = np.mean(v, axis=0)
    b = np.tile(b,(32,1))
    mean_video_feature[idx] = b

pickle.dump(mean_video_feature, open('data/mean_video_feature.p', 'wb'))
