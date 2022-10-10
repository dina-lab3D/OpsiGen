import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim


def grade_transformation(pts1, pts2, mat, descs_dist):
    pts_dist = F.normalize(torch.cdist(pts1 @ mat.T, pts2), dim=0)

    return torch.sum(pts_dist * descs_dist)


def find_transformation(pts1, pts2, descs1, descs2):
    descs_dist = torch.cdist(torch.tensor(descs1), torch.tensor(descs2))
    mat = torch.ones((3, 3), requires_grad=True)

    optimizer = optim.Adam([mat], lr=0.1)
    while True:
        optimizer.zero_grad()
        loss = grade_transformation(torch.tensor(pts1, dtype=torch.float32), torch.tensor(pts2, dtype=torch.float32),
                                    F.normalize(mat, dim=0), descs_dist)
        loss.backward()
        optimizer.step()
        print(loss.item())
        print(mat)
