import time
from math import floor
import numpy as np
import cv2
from scipy.sparse import csr_matrix
import util_sweep


def compute_photometric_stereo_impl(lights, images):
    """
    Given a set of images taken from the same viewpoint and a corresponding set
    of directions for light sources, this function computes the albedo and
    normal map of a Lambertian scene.

    If the computed albedo for a pixel has an L2 norm less than 1e-7, then set
    the albedo to black and set the normal to the 0 vector.

    Normals should be unit vectors.

    Input:
        lights -- N x 3 array.  Rows are normalized and are to be interpreted
                  as lighting directions.
        images -- list of N images.  Each image is of the same scene from the
                  same viewpoint, but under the lighting condition specified in
                  lights.
    Output:
        albedo -- float32 image. When the input 'images' are RGB, it should be of dimension height x width x 3,
                  while in the case of grayscale 'images', the dimension should be height x width x 1.
        normals -- float32 height x width x 3 image with dimensions matching
                   the input images.
    """
    start = time.time()
    # I = kd N * L --> I = L*G --> G = (LT L)^-1 (LT I)
    # kd = ||G||
    # N = 1/kd * G
    img_np_mat = np.array(images)
    if len(images[0].shape) == 2:
        img_np_mat = np.expand_dims(img_np_mat, axis=2)
    N, height, width, color = img_np_mat.shape
    G_1 = np.linalg.inv(np.matmul(np.transpose(lights), lights))
    img_np_mat = np.reshape(img_np_mat, (N, -1))
    G_2 = np.matmul(G_1, np.transpose(lights))
    # 3 x height x width x 3 or 3 x height x width x 1
    G = np.matmul(G_2, img_np_mat)
    G = np.reshape(G, (3, height, width, color))
    albedo = np.linalg.norm(G, axis=0)  # height x width x color
    # albedo[albedo==0] = np.inf
    # div = np.expand_dims(albedo[:,:,0], axis = 2)
    # div = np.concatenate((div,div,div), axis=2)

    normals = np.transpose(G[:, :, :, 0], (1, 2, 0))
    div = albedo
    if color == 1:
        div = np.concatenate((div, div, div), axis=2)
    normals[div != 0] = normals[div != 0] / div[div != 0]

    mask = np.linalg.norm(albedo, axis=2) < 1e-7
    mask = np.expand_dims(mask, axis=2)
    if color == 1:
        albedo[mask] = 0
    mask = np.concatenate((mask, mask, mask), axis=2)
    normals[mask] = 0
    norm = np.expand_dims(np.linalg.norm(normals, axis=2), axis=2)
    norm = np.concatenate((norm, norm, norm), axis=2)
    normals[norm != 0] = normals[norm != 0] / norm[norm != 0]

    done = time.time()
    elapsed = done - start
    print("Time taken for compute_photometric_stereo_impl is %.5f sec" % elapsed)

    return albedo, normals
    



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
    height, width, _ = points.shape
    points = np.concatenate(
        (points, np.ones((points.shape[0], points.shape[1], 1))), axis=2)
    points = np.reshape(points, (-1, 4))
    print(points.shape)
    print(K.shape)
    projections_3D = np.matmul(np.matmul(K, Rt), np.transpose(points))
    projections_3D = np.reshape(projections_3D, (3, height, width))
    projections_3D = np.transpose(projections_3D, (1, 2, 0))
    projections = projections_3D[:, :, 0:2]
    z = projections_3D[:, :, 2]
    z[z == 0] = np.inf
    projections = projections / np.expand_dims(z, axis=2)
    return projections


def preprocess_ncc_impl(image, ncc_size):
    """
    Prepare normalized patch vectors according to normalized cross
    correlation.

    This is a preprocessing step for the NCC pipeline.  It is expected that
    'preprocess_ncc' is called on every input image to preprocess the NCC
    vectors and then 'compute_ncc' is called to compute the dot product
    between these vectors in two images.

    NCC preprocessing has two steps.
    (1) Compute and subtract the mean.
    (2) Normalize the vector.

    The mean is per channel.  i.e. For an RGB image, over the ncc_size**2
    patch, compute the R, G, and B means separately.  The normalization
    is over all channels.  i.e. For an RGB image, after subtracting out the
    RGB mean, compute the norm over the entire (ncc_size**2 * channels)
    vector and divide.

    If the norm of the vector is < 1e-6, then set the entire vector for that
    patch to zero.

    Patches that extend past the boundary of the input image at all should be
    considered zero.  Their entire vector should be set to 0.

    Patches are to be flattened into vectors with the default numpy row
    major order.  For example, given the following
    2 (height) x 2 (width) x 2 (channels) patch, here is how the output
    vector should be arranged.

    channel1         channel2
    +------+------+  +------+------+ height
    | x111 | x121 |  | x112 | x122 |  |
    +------+------+  +------+------+  |
    | x211 | x221 |  | x212 | x222 |  |
    +------+------+  +------+------+  v
    width ------->

    v = [ x111, x121, x211, x221, x112, x122, x212, x222 ]

    see order argument in np.reshape

    Input:
        image -- height x width x channels image of type float32
        ncc_size -- integer width and height of NCC patch region; assumed to be odd
    Output:
        normalized -- heigth x width x (channels * ncc_size**2) array
    """
    if len(image.shape) ==2:
        image = np.expand_dims(image, axis = 2)
    height, width, channels = image.shape
    pad_idx = int((ncc_size -1) / 2)
    padded_img = np.zeros((height + pad_idx*2, width + pad_idx*2, channels), dtype=float)
    padded_img[pad_idx:-pad_idx, pad_idx:-pad_idx, :] = image
    normalized = np.zeros((height, width, channels*ncc_size ** 2))
    for i in range(height):
        for j in range(width):
            patch = padded_img[i:i+ncc_size, j:j+ncc_size,:]
            patch =np.reshape(patch, (ncc_size*ncc_size, channels))
            patch = patch - np.mean(patch, axis = 0)
            patch = np.reshape(np.transpose(patch), ncc_size * ncc_size *channels)
            norm = np.linalg.norm(patch)
            if i<pad_idx or i>height-pad_idx-1 or j<pad_idx or j>width-pad_idx-1:
                normalized[i, j, :] = np.zeros(ncc_size * ncc_size * channels)
            else:
                if norm < 1e-6:
                    normalized[i, j, :] = np.zeros(ncc_size * ncc_size * channels)
                else:
                    patch = patch / norm
                    normalized[i, j, :] = patch[:]
    return normalized


def compute_ncc_impl(image1, image2):
    """
    Compute normalized cross correlation between two images that already have
    normalized vectors computed for each pixel with preprocess_ncc.

    Input:
        image1 -- height x width x (channels * ncc_size**2) array
        image2 -- height x width x (channels * ncc_size**2) array
    Output:
        ncc -- height x width normalized cross correlation between image1 and
               image2.
    """
    ncc = image1 * image2
    ncc = np.sum(ncc, axis = 2)
    return ncc
