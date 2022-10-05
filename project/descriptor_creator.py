import mrcfile
import numpy as np
import pickle
import models
import torch
from skimage.feature import match_descriptors
import math
import time


class DescriptorCreator:
    def __init__(self, map1_path, map2_path, model_path, threas):
        with mrcfile.open(map1_path) as protein:
            self.map1 = protein.data
        with mrcfile.open(map2_path) as protein:
            self.map2 = protein.data

        self.threas = threas

        self.model = pickle.load(open(model_path, 'rb'))
        self.preprocess()

    @staticmethod
    def devide_to_patches(map, threas):
        x_splitted = np.split(map, map.shape[0] / (2 * threas), axis=0)
        xy_splitted = [np.split(x_map, x_map.shape[1] / (2 * threas), axis=1) for x_map in x_splitted]
        xyz_splitted = [[np.split(xy_map, xy_map.shape[2] / (2 * threas), axis=2) for xy_map in y_list]
                        for y_list in xy_splitted]

        return xyz_splitted

    @staticmethod
    def volume(shape):
        volume = 1
        for i in shape:
            volume *= i

        return volume

    def reshape(self, shape, arr):
        return torch.tensor(arr).unsqueeze(dim=1)

    @staticmethod
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    @staticmethod
    def generate_points(map, threas, grid_density):

        x_length = map.shape[0]
        y_length = map.shape[1]
        z_length = map.shape[2]

        x = np.linspace(0 + threas, x_length - threas, int(x_length / grid_density)).astype(np.int)
        y = np.linspace(0 + threas, x_length - threas, int(x_length / grid_density)).astype(np.int)
        z = np.linspace(0 + threas, x_length - threas, int(x_length / grid_density)).astype(np.int)

        points = np.array(DescriptorCreator.cartesian_product(x, y, z))

        counts, vals = np.histogram(map, bins=50)
        t = vals[45]

        res = {}
        for i, point in enumerate(points):
            patch = DescriptorCreator.cut_from_map(map, point, threas)
            if np.max(patch) > t:
                res[i] = patch

        return points[list(res.keys())], list(res.values())

    @staticmethod
    def cut_from_map(map, point, threas):
        res = map[point[0] - threas:point[0] + threas, point[1] - threas: point[1] + threas,
              point[2] - threas:point[2] + threas]
        return res

    def generate_descriptors(self, map, grid_density):
        points, patches = self.generate_points(map, self.threas, grid_density)
        map_descs = self.reshape(map.shape, np.array(patches))
        bad_indexes = []
        for i, desc in enumerate(map_descs.squeeze()):
            if np.var(desc.detach().numpy()) < (10 ** (-4)):
                bad_indexes.append(i)
        model_descs = self.model(map_descs).squeeze().detach().numpy()
        vol = self.volume(model_descs.shape)
        final_descs = model_descs.reshape(model_descs.shape[0], int(vol / model_descs.shape[0]))

        return bad_indexes, final_descs

    @staticmethod
    def calculate_correlation(points1, points2, rotation, translation, descs1, descs2):
        points2_new = points1 @ rotation + translation
        matches = match_descriptors(points2_new, points2, max_distance=20)

        if matches.shape[0] == 0:
            return math.inf

        kp1 = np.array(descs1[matches[:, 0]])
        kp2 = np.array(descs2[matches[:, 1]])
        print(len(matches))

        return np.linalg.norm(kp1 - kp2)

    @staticmethod
    def ransac(x, y, descs_x, descs_y, funcDist, minPtNum, iterNum, thDist, thInlrRatio):
        """
        Use RANdom SAmple Consensus to find a fit from X to Y.
        :param x: M*n matrix including n points with dim M
        :param y: N*n matrix including n points with dim N
        :param funcFindF: a function with interface f1 = funcFindF(x1,y1) where:
                    x1 is M*n1
                    y1 is N*n1 n1 >= minPtNum
                    f is an estimated transformation between x1 and y1 - can be of any type
        :param funcDist: a function with interface d = funcDist(x1,y1,f) that uses f returned by funcFindF and returns the
                    distance between <x1 transformed by f> and <y1>. d is 1*n1.
                    For line fitting, it should calculate the distance between the line and the points [x1;y1];
                    For homography, it should project x1 to y2 then calculate the dist between y1 and y2.
        :param minPtNum: the minimum number of points with whom can we find a fit. For line fitting, it's 2. For
                    homography, it's 4.
        :param iterNum: number of iterations (== number of times we draw a random sample from the points
        :param thDist: inlier distance threshold.
        :param thInlrRatio: ROUND(THINLRRATIO*n) is the inlier number threshold
        :return: [f, inlierIdx] where: f is the fit and inlierIdx are the indices of inliers
        """

        ptNum = len(x)
        thInlr = round(thInlrRatio * ptNum)

        best_dist = math.inf
        counter = 0
        small_dist_th = 0.05
        for i in range(iterNum):
            counter += 1
            permut1 = np.random.permutation(ptNum)
            permut2 = np.random.permutation(ptNum)
            sampleIdx1 = permut1[range(minPtNum)]
            sampleIdx2 = permut2[range(minPtNum)]
            rotation, translation = DescriptorCreator.calcPointBasedReg(x[sampleIdx1, :], y[sampleIdx2, :])
            small_dist = funcDist(x[sampleIdx1, :], y[sampleIdx2, :], rotation,
                                  translation, descs_x[sampleIdx1, :], descs_y[sampleIdx2, :])
            while small_dist > small_dist_th:
                permut1 = np.random.permutation(ptNum)
                permut2 = np.random.permutation(ptNum)
                sampleIdx1 = permut1[range(minPtNum)]
                sampleIdx2 = permut2[range(minPtNum)]
                rotation, translation = DescriptorCreator.calcPointBasedReg(x[sampleIdx1, :], y[sampleIdx2, :])
                small_dist = funcDist(x[sampleIdx1, :], y[sampleIdx2, :], rotation,
                                      translation, descs_x[sampleIdx1, :], descs_y[sampleIdx2, :])

            dist = funcDist(x, y, rotation, translation, descs_x, descs_y)
            if dist < best_dist:
                best_dist = dist
                print("Found better:")
                print("Dist is :", dist)
                print(rotation)
                best_func = rotation, translation

            print(counter)


        return best_func

    @staticmethod
    def index_to_centroid(shape, threas, index):
        diameter = threas * 2
        x_length = shape[0] / diameter
        y_length = shape[1] / diameter
        z_length = shape[2] / diameter

        x = int(index / (y_length * z_length))
        y = int((index - (y_length * z_length) * x) / y_length)
        z = int(index % z_length)

        return x * diameter + threas, y * diameter + threas, z * diameter + threas

    @staticmethod
    def calcPointBasedReg(x, y):
        centroid1 = x.mean(axis=0)
        centroid2 = y.mean(axis=0)

        centered_XPoints = x - centroid1
        centered_YPoints = y - centroid2

        sigma = centered_YPoints.T @ centered_XPoints
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

        return rotation, translation

    def preprocess(self):
        bad_indexes1, descs1 = self.generate_descriptors(self.map1, int(self.threas / 3))
        bad_indexes2, descs2 = self.generate_descriptors(self.map2, int(self.threas / 3))

        good_indexes = [i for i in range(len(descs1)) if i not in list(set(bad_indexes1 + bad_indexes2))]

        points1 = np.array(
            [DescriptorCreator.index_to_centroid(self.map1.shape, self.threas, index) for index in good_indexes])
        points2 = np.array(
            [DescriptorCreator.index_to_centroid(self.map2.shape, self.threas, index) for index in good_indexes])

        # matches = match_descriptors(descs1[good_indexes, :], descs2[good_indexes, :])
        #
        # kp1 = points1[matches[:, 0]]
        # kp2 = points2[matches[:, 1]]

        rot, trans = self.ransac(points1, points2, descs1, descs2, DescriptorCreator.calculate_correlation, 3, 1000, 60, 0.0001)
        print(rot, trans)

        # self.ransac(descs1, descs2)
