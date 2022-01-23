import numpy as np
from util import preprocess_ncc, compute_ncc, project, unproject_corners, \
    pyrdown, pyrup, compute_photometric_stereo

def project_impl(K, Rt, points):
    """
    Project 3D points into a calibrated camera.
    Input:
        K -- camera intrinsics calibration matrix
        Rt -- 3 x 4 camera extrinsics calibration matrix
        points -- height x width x 3 array of 3D points
    Output:
        projections -- height x width x 2 array of 2D projections
    """
    # k3 = np.transpose(K[:, 2])
    # r3x3 = Rt[:, :3]
    #
    # pts_size = points.shape
    # proj = np.zeros((pts_size[0], pts_size[1], 2))
    # for i in range(pts_size[0]):
    #     for j in range(pts_size[1]):
    #         pt = points[i, j, :]
    #         temp1 = np.dot(r3x3, pt)
    #         temp2 = temp1 + k3
    #         proj[i, j] = temp2[:2]
    # return proj
    points = np.concatenate((points, np.ones((points.shape[0], points.shape[1], 1))), axis=2)
    print(K.shape)

    projections_3D = np.matmul(np.matmul(K, Rt), np.transpose(points, (2, 0, 1)))
    projections_3D = np.transpose(projections_3D, (1, 2, 0))
    projections = projections_3D[:, :, 0:2]
    z = projections_3D[:, :, 2]
    z[z == 0] = np.inf
    projections = projections / np.expand_dims(z, axis=2)
    return projections


width = 20
height = 10
f = 1

K = np.array((
    (f, 0, width / 2.0),
    (0, f, height / 2.0),
    (0, 0, 1)
))

A = np.random.random((3, 3))
U, S, Vt = np.linalg.svd(A)
R = U.dot(Vt)

Rt = np.zeros((3, 4), dtype=np.float32)
Rt[:, :3] = R
Rt[:, 3] = np.random.random(3)

depth = 2
point = unproject_corners(K, width, height, depth, Rt)

projection = project(K, Rt, point)

print(point.shape)
print(type(point))
