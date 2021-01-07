from keras.losses import binary_crossentropy
import keras.backend as K
import tensorflow as tf 

# https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py

epsilon = 1e-5
smooth = 1

def dice_loss(y_true, y_pred, smooth = 1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    loss = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    return loss

def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg) 
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall

def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 

def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.1
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1. - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)

# -------------------------------------------------------------------------------------------------

# https://github.com/umbertogriffo/focal-loss-keras/blob/master/src/loss_function/losses.py

def binary_focal_loss_fixed(y_true, y_pred, gamma=2., alpha=.25):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

# ----------------------------------------------------------------------------------------------------------------------------
    
# https://stackoverflow.com/questions/60609585/good-performance-with-accuracy-but-not-with-dice-loss-in-image-segmentation

def soft_dice_loss(y_true, y_pred, epsilon=1e-6): 
    """Soft dice loss calculation for arbitrary batch size, number of classes, and number of spatial dimensions.
    Assumes the `channels_last` format.
  
    # Arguments
        y_true: b x X x Y( x Z...) x c One hot encoding of ground truth
        y_pred: b x X x Y( x Z...) x c Network output, must sum to 1 over c channel (such as after softmax) 
        epsilon: Used for numerical stability to avoid divide by zero errors
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
   
    # skip the batch and class axis for calculating Dice score
    axes = tuple(range(1, len(y_pred_f.shape)-1)) 
    numerator = 2. * K.sum(y_pred_f * y_true_f, axes)
    denominator = K.sum(K.square(y_pred_f) + K.square(y_true_f), axes)
    
    return 1 - K.mean(numerator / (denominator + epsilon)) # average over classes and batch

#---------------------------------------------------------------------------------------------
    
# https://github.com/sizhuoli/Weighted-Tversky-loss/blob/master/Weighted_Tversky_loss.py

def weighted_tversky(y_true, y_pred, alpha=0.6, beta=0.4):
    """compute the weighted Tversky loss with weight maps"""
    #annotation
    y_t = y_true[...,0]
    y_t = y_t[...,np.newaxis]
    #weights
    y_weights = y_true[...,1]
    y_weights = y_weights[...,np.newaxis]
    ones = K.ones(K.shape(y_t))
    #p0: prob that the pixel is of class 1
    p0 = y_pred  
    #p1: prob that the pixel is of class 0
    p1 = ones - y_pred  
    g0 = y_t
    g1 = ones - y_t
    #terms in the Tversky loss function combined with weights
    tp = tf.reduce_sum(y_weights * p0 * g0)
    fp = alpha * tf.reduce_sum(y_weights * p0 * g1)
    fn = beta * tf.reduce_sum(y_weights * p1 * g0)
    #add to the denominator a small epsilon to prevent the value from being undefined 
    EPS = 1e-5
    num = tp
    den = tp + fp + fn + EPS
    result = num / den
    return 1 - tf.reduce_mean(result)


#---------------------------------------------------------------------------------------------
    
# https://github.com/LIVIAETS/boundary-loss/blob/master/keras_loss.py

from scipy.ndimage import distance_transform_edt as distance

def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res


def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)


def surface_loss_keras(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)

#---------------------------------------------------------------------------------------------
    
# https://github.com/danielenricocahall/Keras-Weighted-Hausdorff-Distance-Loss/blob/master/hausdorff/hausdorff.py

import tensorflow as tf
import keras.backend as K
import numpy as np
from sklearn.utils.extmath import cartesian
import math


def cdist(A, B):
    """
    Computes the pairwise Euclidean distance matrix between two tensorflow matrices A & B, similiar to scikit-learn cdist.
    For example:
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2],
         [3, 4]]
    should return:
        [[0, 2.82],
         [2.82, 0]]
    :param A: m_a x n matrix
    :param B: m_b x n matrix
    :return: euclidean distance matrix (m_a x m_b)
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D


def weighted_hausdorff_distance(w, h, alpha):
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w), np.arange(h)]), dtype=tf.float32)
    max_dist = math.sqrt(w ** 2 + h ** 2)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            eps = 1e-6
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss


#---------------------------------------------------------------------------------------------
    
# https://github.com/N0vel/weighted-hausdorff-distance-tensorflow-keras-loss/blob/master/weighted_hausdorff_loss.py

import math

import numpy as np
import tensorflow as tf
from sklearn.utils.extmath import cartesian

resized_height = 144  
resized_width  = 256
max_dist = math.sqrt(resized_height**2 + resized_width**2)
n_pixels = resized_height * resized_width
all_img_locations = tf.convert_to_tensor(cartesian([np.arange(resized_height), np.arange(resized_width)]),
                                                   tf.float32)


def tf_repeat(tensor, repeats):
    """
    Args:
    input: A Tensor. 1-D or higher.
    repeats: A list. Number of repeat for each dimension, length must be the same as the number of dimensions in input
    Returns:
    
    A Tensor. Has the same type as input. Has the shape of tensor.shape * repeats
    """
    with tf.variable_scope("repeat"):
        expanded_tensor = tf.expand_dims(tensor, -1)
        multiples = [1] + repeats
        tiled_tensor = tf.tile(expanded_tensor, multiples = multiples)
        repeated_tesnor = tf.reshape(tiled_tensor, tf.shape(tensor) * repeats)
    return repeated_tesnor



def Weighted_Hausdorff_loss(y_true, y_pred):
    # https://arxiv.org/pdf/1806.07564.pdf
    #prob_map_b - y_pred
    #gt_b - y_true

    terms_1 = []
    terms_2 = []
    y_true = tf.squeeze(y_true, axis=-1)
    y_pred = tf.squeeze(y_pred, axis=-1)
#     y_true = tf.reduce_mean(y_true, axis=-1)
#     y_pred = tf.reduce_mean(y_pred, axis=-1)
    for b in range(batch_size):
        gt_b = y_true[b]
        prob_map_b = y_pred[b]
        # Pairwise distances between all possible locations and the GTed locations
        n_gt_pts = tf.reduce_sum(gt_b)
        gt_b = tf.where(tf.cast(gt_b, tf.bool))
        gt_b = tf.cast(gt_b, tf.float32)
        d_matrix = tf.sqrt(tf.maximum(tf.reshape(tf.reduce_sum(gt_b*gt_b, axis=1), (-1, 1)) + tf.reduce_sum(all_img_locations*all_img_locations, axis=1)-2*(tf.matmul(gt_b, tf.transpose(all_img_locations))), 0.0))
        d_matrix = tf.transpose(d_matrix)
        # Reshape probability map as a long column vector,
        # and prepare it for multiplication
        p = tf.reshape(prob_map_b, (n_pixels, 1))
        n_est_pts = tf.reduce_sum(p)
        p_replicated = tf_repeat(tf.reshape(p, (-1, 1)), [1, n_gt_pts])
        eps = 1e-6
        alpha = 4
        # Weighted Hausdorff Distance
        term_1 = (1 / (n_est_pts + eps)) * tf.reduce_sum(p * tf.reshape(tf.reduce_min(d_matrix, axis=1), (-1, 1)))
        d_div_p = tf.reduce_min((d_matrix + eps) / (p_replicated**alpha + eps / max_dist), axis=0)
        d_div_p = tf.clip_by_value(d_div_p, 0, max_dist)
        term_2 = tf.reduce_mean(d_div_p, axis=0)
        terms_1.append(term_1)
        terms_2.append(term_2)
    terms_1 = tf.stack(terms_1)
    terms_2 = tf.stack(terms_2)
    terms_1 = tf.Print(tf.reduce_mean(terms_1), [tf.reduce_mean(terms_1)], "term 1")
    terms_2 = tf.Print(tf.reduce_mean(terms_2), [tf.reduce_mean(terms_2)], "term 2")
    res = terms_1 + terms_2
    return res