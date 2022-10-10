import mrcfile
import numpy as np
import pickle
import models
import torch
from MapSeq import MapSeq
from skimage.feature import match_descriptors
import math
import mrcfile

import transformation_finder

CRYO_FILE_TEMPLATE_A = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/a{}.mrc"
CRYO_FILE_TEMPLATE_B = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/b{}.mrc"
CRYO_FILE_TEMPLATE_C = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/c{}.mrc"
CRYO_FILE_TEMPLATE_D = "/mnt/c/Users/zlils/Documents/university/biology/cryo-folding/cryo-em-data/d{}.mrc"


def is_empty_point(arr, threashold_for_density = 500):
    volume = 1
    for index in range(len(arr.shape)):
        volume *= arr.shape[index]
    # print("Above the mean values", np.sum(np.where(arr.detach().numpy() > np.mean(arr.detach().numpy()))))
    if np.sum(np.where(arr > np.mean(arr))) < volume / 4:
        return True
    return False


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


    @staticmethod
    def cartesian_product(*arrays):
        la = len(arrays)
        dtype = np.result_type(*arrays)
        arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*arrays)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    @staticmethod
    def generate_patches(map, points, threas):
        res = {}
        for i, point in enumerate(points):
            patch = DescriptorCreator.cut_from_map(map, point, threas)
            if not is_empty_point(patch):
                # mrcfile.write(CRYO_FILE_TEMPLATE_A, patch, overwrite=True)
                res[i] = patch

        return points[list(res.keys())], list(res.values())

    @staticmethod
    def cut_from_map(map, point, threas):
        unshaped_desc = map[point[0] - threas:point[0] + threas, point[1] - threas: point[1] + threas,
              point[2] - threas:point[2] + threas]

        desc = MapSeq._to_shape(unshaped_desc, (threas * 2, threas * 2, threas * 2))
        return desc

    def generate_descriptors(self, map, threas):
        edges = MapSeq.find_edges(map)
        chosen_points = edges[np.all(edges % int(self.threas / 2) == 0, axis=1)]
        points, patches = self.generate_patches(map, chosen_points, self.threas)
        map_descs = torch.tensor(np.array(patches)).unsqueeze(dim=1)
        model_descs = self.model(map_descs).squeeze().detach().numpy()

        return patches, model_descs, points

    @staticmethod
    def calculate_correlation(points1, points2, descs1, descs2, rotation, translation):
        points2_new = (rotation @ points1.T + translation).T
        matches = match_descriptors(points2_new, points2, max_distance=15)

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

        ptNum = x.shape[0]
        best_correlation = math.inf
        best_func = np.eye(3), np.ones(3)
        for i in range(iterNum):
            permut = np.random.permutation(ptNum)
            sampleIdx = permut[range(minPtNum)]
            func = DescriptorCreator.calcPointBasedReg(x[sampleIdx,:],y[sampleIdx,:])
            if func is None:
                continue
            else:
                rotation, translation = func
            correlation = funcDist(x, y, descs_x, descs_y, rotation, translation)
            # b = (dist <= thDist)
            # r = np.array(range(b.shape[0]))
            # inlier1 = r[b]
            # inlrNum[i] = len(inlier1)
            if correlation < best_correlation:
                best_correlation = correlation
                best_func = rotation, translation
                print("New best function:", rotation, translation)
                continue

        return best_func

    # @staticmethod
    # def index_to_centroid(shape, threas, index):
    #     diameter = threas * 2
    #     x_length = shape[0] / diameter
    #     y_length = shape[1] / diameter
    #     z_length = shape[2] / diameter
    #
    #     x = int(index / (y_length * z_length))
    #     y = int((index - (y_length * z_length) * x) / y_length)
    #     z = int(index % z_length)
    #
    #     return x * diameter + threas, y * diameter + threas, z * diameter + threas

    @staticmethod
    def calcPointBasedReg(A, B):
        A = A.T
        B = B.T
        assert A.shape == B.shape

        num_rows, num_cols = A.shape
        if num_rows != 3:
            raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

        num_rows, num_cols = B.shape
        if num_rows != 3:
            raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

        # find mean column wise
        centroid_A = np.mean(A, axis=1)
        centroid_B = np.mean(B, axis=1)

        # ensure centroids are 3x1
        centroid_A = centroid_A.reshape(-1, 1)
        centroid_B = centroid_B.reshape(-1, 1)

        # subtract mean
        Am = A - centroid_A
        Bm = B - centroid_B

        H = Am @ np.transpose(Bm)

        # find rotation
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # special reflection case
        if np.linalg.det(R) < 0:
            return None

        t = -R @ centroid_A + centroid_B

        return R, t

    def preprocess(self):
        # edges1 = MapSeq.find_edges(self.map1)
        # edges2 = MapSeq.find_edges(self.map2)
        # chosen_points1 = edges1[np.all(edges1 % int(self.threas / 4) == 0, axis=1)]
        # chosen_points2 = edges2[np.all(edges2 % int(self.threas / 4) == 0, axis=1)]
        patches1, descs1, points1 = self.generate_descriptors(self.map1, self.threas)
        patches2, descs2, points2 = self.generate_descriptors(self.map2, self.threas)

        # points1 = points1[[i for i in range(len(descs1)) if i not in list(set(bad_indexes1))]]
        # points2 = points2[[i for i in range(len(descs2)) if i not in list(set(bad_indexes2))]]

        # points1 = np.array(
        #     [DescriptorCreator.index_to_centroid(self.map1.shape, self.threas, index) for index in good_indexes])
        # points2 = np.array(
        #     [DescriptorCreator.index_to_centroid(self.map2.shape, self.threas, index) for index in good_indexes])

        matches = match_descriptors(descs1, descs2, cross_check=True)
        #
        kp1 = points1[matches[:, 0]]
        kp2 = points2[matches[:, 1]]
        for i in range(len(matches)):
            # mrcfile.write(CRYO_FILE_TEMPLATE_A, patches1[matches[0][0]], overwrite=True)
            # mrcfile.write(CRYO_FILE_TEMPLATE_B, patches2[matches[0][1]], overwrite=True)
            print(np.linalg.norm(kp1[i] - kp2[i]))

        breakpoint()
        # transformation_finder.find_transformation(points1, points2, descs1, descs2)
        # mrcfile.write(CRYO_FILE_TEMPLATE_A, descs1[matches[0][0]], overwrite=True)
        # mrcfile.write(CRYO_FILE_TEMPLATE_B, descs2[matches[0][1]], overwrite=True)

        # rot, trans = self.ransac(points1, points2, descs1, descs2, DescriptorCreator.calculate_correlation, 3, 10000, 60, 0.0001)
        # print(rot, trans)

        # self.ransac(descs1, descs2)
