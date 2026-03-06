import os

# for visualizations
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np # for using np arrays
import cv2
import time


# for bulding and running deep learning model
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import concatenate
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
from sklearn.model_selection import train_test_split

def LoadData (path1, path2):
    """
    Looks for relevant filenames in the shared path
    Returns 2 lists for original and masked files respectively

    """
    # Read the images folder like a list
    image_visible_dataset = os.listdir(path1)
    image_infra_dataset = os.listdir(path2)

    # Make a list for images and masks filenames
    visible_img = []
    infra_img = []
    for file in image_visible_dataset:
        visible_img.append(file)
    for file in image_infra_dataset:
        infra_img.append(file)

    #We should have the name of the images taken as imgX or visibleX or infraX, with the infra and visible images named with the same number.
    visible_img.sort()
    infra_img.sort()

    return visible_img, infra_img



def EncoderMiniBlock(inputs, n_filters=32, dropout_prob=0.3, max_pooling=True):
    """
    This block uses multiple convolution layers, max pool, relu activation to create an architecture for learning.
    Dropout can be added for regularization to prevent overfitting.
    The block returns the activation values for next layer along with a skip connection which will be used in the decoder
    """
    # Add 2 Conv Layers with relu activation and HeNormal initialization using TensorFlow
    # Proper initialization prevents from the problem of exploding and vanishing gradients
    # 'Same' padding will pad the input to conv layer such that the output has the same height and width (hence, is not reduced in size)
    conv = Conv2D(n_filters,
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(inputs)
    conv = Conv2D(n_filters,
                  3,   # Kernel size
                  activation='relu',
                  padding='same',
                  kernel_initializer='HeNormal')(conv)

    # Batch Normalization will normalize the output of the last layer based on the batch's mean and standard deviation
    conv = BatchNormalization()(conv, training=False)

    # In case of overfitting, dropout will regularize the loss and gradient computation to shrink the influence of weights on output
    if dropout_prob > 0:
        conv = tf.keras.layers.Dropout(dropout_prob)(conv)

    # Pooling reduces the size of the image while keeping the number of channels same
    # Pooling has been kept as optional as the last encoder layer does not use pooling (hence, makes the encoder block flexible to use)
    # Below, Max pooling considers the maximum of the input slice for output computation and uses stride of 2 to traverse across input image
    if max_pooling:
        next_layer = tf.keras.layers.MaxPooling2D(pool_size = (2,2))(conv)
    else:
        next_layer = conv

    # skip connection (without max pooling) will be input to the decoder layer to prevent information loss during transpose convolutions
    skip_connection = conv

    return next_layer, skip_connection



def DecoderMiniBlock(prev_layer_input, skip_layer_input, n_filters=32):
    """
    Decoder Block first uses transpose convolution to upscale the image to a bigger size and then,
    merges the result with skip layer results from encoder block
    Adding 2 convolutions with 'same' padding helps further increase the depth of the network for better predictions
    The function returns the decoded layer output
    """
    # Start with a transpose convolution layer to first increase the size of the image
    up = Conv2DTranspose(
                 n_filters,
                 (3,3),    # Kernel size
                 strides=(2,2),
                 padding='same')(prev_layer_input)

    # Merge the skip connection from previous block to prevent information loss
    merge = concatenate([up, skip_layer_input], axis=3)

    # Add 2 Conv Layers with relu activation and HeNormal initialization for further processing
    # The parameters for the function are similar to encoder
    conv = Conv2D(n_filters,
                 3,     # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(merge)
    conv = Conv2D(n_filters,
                 3,   # Kernel size
                 activation='relu',
                 padding='same',
                 kernel_initializer='HeNormal')(conv)
    return conv


def UNetCompiled(input_size=(1296, 1032, 1), n_filters=32, n_classes=2): 
    """ U-Net adapté pour les images IR complètes. Input: image IR complète (H=1296, W=1032, C=1) Output: champs de déformation (dx, dy) avec n_classes=2 """ 
    inputs = Input(input_size)
    # Encoder 
    cblock1 = EncoderMiniBlock(inputs, n_filters, dropout_prob=0, max_pooling=True) 
    cblock2 = EncoderMiniBlock(cblock1[0], n_filters*2, dropout_prob=0, max_pooling=True) 
    cblock3 = EncoderMiniBlock(cblock2[0], n_filters*4, dropout_prob=0, max_pooling=True) 
    cblock4 = EncoderMiniBlock(cblock3[0], n_filters*8, dropout_prob=0.3, max_pooling=True) 
    cblock5 = EncoderMiniBlock(cblock4[0], n_filters*16, dropout_prob=0.3, max_pooling=False) 
    
    # Decoder 
    ublock6 = DecoderMiniBlock(cblock5[0], cblock4[1], n_filters*8) 
    ublock7 = DecoderMiniBlock(ublock6, cblock3[1], n_filters*4) 
    ublock8 = DecoderMiniBlock(ublock7, cblock2[1], n_filters*2) 
    ublock9 = DecoderMiniBlock(ublock8, cblock1[1], n_filters) 
    # Output 
    conv9 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock9) 
    conv10 = Conv2D(n_classes, 1, padding='same')(conv9) 
    model = tf.keras.Model(inputs=inputs, outputs=conv10) 
    return model



def image_generator(file_list_vis, file_list_ir, path_vis, path_infra, patch_size_vis, patch_size_ir):
    for vis_name, ir_name in zip(file_list_vis, file_list_ir):
        vis_img = np.array(Image.open(os.path.join(path_vis, vis_name)).convert('RGB'), dtype=np.float32)/255.0
        ir_img  = np.array(Image.open(os.path.join(path_infra, ir_name)).convert('L'), dtype=np.float32)/255.0
        ir_img  = np.expand_dims(ir_img, axis=-1)


        yield ir_img, vis_img
    



@tf.function
def stn_warp(image, flow_field):
    """
    Warping amélioré avec meilleure interpolation
    """
    batch_size = tf.shape(image)[0]
    height = tf.shape(image)[1]
    width = tf.shape(image)[2]
    
    # Créer grille normalisée [-1, 1]
    y_coords = tf.linspace(-1.0, 1.0, height)
    x_coords = tf.linspace(-1.0, 1.0, width)
    x_grid, y_grid = tf.meshgrid(x_coords, y_coords)
    
    # Normaliser le flow field (important!)
    # On suppose que flow_field est en pixels, on le normalise
    flow_x = flow_field[0, :, :, 0] / tf.cast(width, tf.float32) * 2.0
    flow_y = flow_field[0, :, :, 1] / tf.cast(height, tf.float32) * 2.0
    
    # Appliquer le flow
    x_new = x_grid + flow_x
    y_new = y_grid + flow_y
    
    # Clip dans [-1, 1]
    x_new = tf.clip_by_value(x_new, -1.0, 1.0)
    y_new = tf.clip_by_value(y_new, -1.0, 1.0)
    
    # Convertir en coordonnées pixel [0, width-1] et [0, height-1]
    x_new_pix = (x_new + 1.0) * tf.cast(width - 1, tf.float32) / 2.0
    y_new_pix = (y_new + 1.0) * tf.cast(height - 1, tf.float32) / 2.0
    
    # Bilinear interpolation
    x0 = tf.cast(tf.floor(x_new_pix), tf.int32)
    x1 = tf.minimum(x0 + 1, width - 1)
    y0 = tf.cast(tf.floor(y_new_pix), tf.int32)
    y1 = tf.minimum(y0 + 1, height - 1)
    
    # Clamp pour sécurité
    x0 = tf.clip_by_value(x0, 0, width - 1)
    x1 = tf.clip_by_value(x1, 0, width - 1)
    y0 = tf.clip_by_value(y0, 0, height - 1)
    y1 = tf.clip_by_value(y1, 0, height - 1)
    
    # Poids d'interpolation
    wx = tf.expand_dims(x_new_pix - tf.cast(x0, tf.float32), axis=-1)
    wy = tf.expand_dims(y_new_pix - tf.cast(y0, tf.float32), axis=-1)
    
    # Gather les 4 coins
    def gather_pixel(y_idx, x_idx):
        indices = tf.stack([y_idx, x_idx], axis=-1)
        return tf.gather_nd(image[0], indices)
    
    Q00 = gather_pixel(y0, x0)
    Q01 = gather_pixel(y0, x1)
    Q10 = gather_pixel(y1, x0)
    Q11 = gather_pixel(y1, x1)
    
    # Interpolation bilinéaire
    top = Q00 * (1.0 - wx) + Q01 * wx
    bottom = Q10 * (1.0 - wx) + Q11 * wx
    interpolated = top * (1.0 - wy) + bottom * wy
    
    return tf.expand_dims(interpolated, 0)

#faire du débug au sein du warp pour voir ce qui ce passe concrètement
#problème gestion des patchs
#donc problème de mirroir en bas.




def sobel_for_loss_ir(img):
    """
    img: Tensor [B, H, W, C] (float32, normalisé)
    return: magnitude Sobel [B, H, W, 1]
    """
    sobel = tf.image.sobel_edges(img)
    # shape: [B, H, W, C, 2] (dx, dy)

    dx = sobel[..., 0]
    dy = sobel[..., 1]

    magnitude = tf.sqrt(dx**2 + dy**2 + 1e-8)
    return magnitude


def ncc_loss(img1, img2):
    """
    Normalized Cross-Correlation loss 
    """
    # Normaliser les images
    img1_mean = tf.reduce_mean(img1)
    img2_mean = tf.reduce_mean(img2)
    
    img1_norm = img1 - img1_mean
    img2_norm = img2 - img2_mean
    
    # Calculer NCC
    numerator = tf.reduce_sum(img1_norm * img2_norm)
    denominator = tf.sqrt(tf.reduce_sum(img1_norm**2) * tf.reduce_sum(img2_norm**2) + 1e-8)
    
    ncc = numerator / denominator
    
    # Retourner 1 - NCC pour minimiser
    return 1.0 - ncc


def gradient_loss(img1, img2):
    """
    Loss sur les gradients
    """
    # Gradients en x et y
    def compute_gradients(img):
        dx = img[:, :, 1:, :] - img[:, :, :-1, :]
        dy = img[:, 1:, :, :] - img[:, :-1, :, :]
        return dx, dy
    
    dx1, dy1 = compute_gradients(img1)
    dx2, dy2 = compute_gradients(img2)
    
    # S'assurer des mêmes dimensions
    min_h = tf.minimum(tf.shape(dx1)[1], tf.shape(dx2)[1])
    min_w = tf.minimum(tf.shape(dx1)[2], tf.shape(dx2)[2])
    
    dx1 = dx1[:, :min_h, :min_w, :]
    dx2 = dx2[:, :min_h, :min_w, :]
    dy1 = dy1[:, :min_h, :min_w, :]
    dy2 = dy2[:, :min_h, :min_w, :]
    
    # L1 loss sur les gradients
    loss_x = tf.reduce_mean(tf.abs(dx1 - dx2))
    loss_y = tf.reduce_mean(tf.abs(dy1 - dy2))
    
    return loss_x + loss_y


def smoothness_loss(flow_field):
    """
    Régularisation pour un flow plus lisse
    """
    # Calculer les différences entre pixels voisins
    dx = flow_field[:, :, 1:, :] - flow_field[:, :, :-1, :]
    dy = flow_field[:, 1:, :, :] - flow_field[:, :-1, :, :]
    
    # L1 norm des différences
    return tf.reduce_mean(tf.abs(dx)) + tf.reduce_mean(tf.abs(dy))




def content_based_crop_with_features(vis_image, ir_image, debug=False, min_matches=10):
    
    vis_np = vis_image.numpy()[0]
    ir_np = ir_image.numpy()[0]
    
    vis_h, vis_w = vis_np.shape[:2]
    ir_h, ir_w = ir_np.shape[:2]
    
    # Convertir en grayscale
    if vis_np.shape[-1] > 1:
        vis_gray = np.mean(vis_np, axis=-1)
    else:
        vis_gray = vis_np[:, :, 0]
    
    if ir_np.shape[-1] > 1:
        ir_gray = np.mean(ir_np, axis=-1)
    else:
        ir_gray = ir_np[:, :, 0]
    
    # Normaliser pour OpenCV
    vis_gray = ((vis_gray - vis_gray.min()) / (vis_gray.max() - vis_gray.min() + 1e-8) * 255).astype(np.uint8)
    ir_gray = ((ir_gray - ir_gray.min()) / (ir_gray.max() - ir_gray.min() + 1e-8) * 255).astype(np.uint8)
    
    # ORB
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    kp_vis, des_vis = orb.detectAndCompute(vis_gray, None)
    kp_ir, des_ir = orb.detectAndCompute(ir_gray, None)
    
    if des_vis is None or des_ir is None or len(kp_vis) < min_matches or len(kp_ir) < min_matches:
        if debug:
            print(f"⚠️ Not enough features detected")
        offset_h = (vis_h - ir_h) // 2
        offset_w = (vis_w - ir_w) // 2
        vis_cropped = tf.image.crop_to_bounding_box(
            vis_image, offset_height=offset_h, offset_width=offset_w,
            target_height=ir_h, target_width=ir_w
        )
        return vis_cropped, ir_image, False
    
    # Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des_vis, des_ir)
    
    if len(matches) < min_matches:
        if debug:
            print(f"⚠️ Not enough matches: {len(matches)}")
        offset_h = (vis_h - ir_h) // 2
        offset_w = (vis_w - ir_w) // 2
        vis_cropped = tf.image.crop_to_bounding_box(
            vis_image, offset_height=offset_h, offset_width=offset_w,
            target_height=ir_h, target_width=ir_w
        )
        return vis_cropped, ir_image, False
    
    matches = sorted(matches, key=lambda x: x.distance)
    num_good = max(min_matches, int(len(matches) * 0.3))
    good_matches = matches[:num_good]
    
    pts_vis = np.float32([kp_vis[m.queryIdx].pt for m in good_matches])
    pts_ir = np.float32([kp_ir[m.trainIdx].pt for m in good_matches])
    
    # CORRECTION: Le problème est ici !
    # pts_vis contient les coordonnées dans VIS (grande image)
    # pts_ir contient les coordonnées dans IR (petite image)
    # On veut trouver où se trouve la zone IR dans VIS
    
    try:
        # Utiliser findHomography plutôt que estimateAffinePartial2D
        # pour mieux gérer les différences de taille
        H_matrix, mask = cv2.findHomography(pts_ir, pts_vis, cv2.RANSAC, 5.0)
        
        if H_matrix is None:
            if debug:
                print("⚠️ Could not estimate homography")
            offset = np.median(pts_vis - pts_ir, axis=0)
            offset_h = int(np.clip(offset[1], 0, vis_h - ir_h))
            offset_w = int(np.clip(offset[0], 0, vis_w - ir_w))
        else:
            # Trouver où les coins de IR se projettent dans VIS
            ir_corners = np.float32([[0, 0], [ir_w, 0], [ir_w, ir_h], [0, ir_h]]).reshape(-1, 1, 2)
            vis_corners = cv2.perspectiveTransform(ir_corners, H_matrix)
            
            # Calculer la bounding box dans VIS
            x_min = int(np.min(vis_corners[:, 0, 0]))
            y_min = int(np.min(vis_corners[:, 0, 1]))
            
            offset_h = int(np.clip(y_min, 0, vis_h - ir_h))
            offset_w = int(np.clip(x_min, 0, vis_w - ir_w))
            
            inliers = np.sum(mask)
            if debug:
                print(f"✓ Inliers: {inliers}/{len(good_matches)} ({inliers/len(good_matches)*100:.1f}%)")
                print(f"✓ Computed bbox in VIS: x={x_min}, y={y_min}")
    except Exception as e:
        if debug:
            print(f"⚠️ Homography failed: {e}")
        offset = np.median(pts_vis, axis=0) - np.array([ir_w/2, ir_h/2])
        offset_h = int(np.clip(offset[1], 0, vis_h - ir_h))
        offset_w = int(np.clip(offset[0], 0, vis_w - ir_w))
    
    if debug:
        print(f"✓ Feature matching - offset_h: {offset_h}, offset_w: {offset_w}, matches: {len(good_matches)}")
    
    # Crop final
    vis_cropped = tf.image.crop_to_bounding_box(
        vis_image,
        offset_height=offset_h,
        offset_width=offset_w,
        target_height=ir_h,
        target_width=ir_w
    )
    
    is_valid = True
    
    return vis_cropped, ir_image, is_valid




def binarize_image(image, threshold=0.5):
    """
    Binarise l'image selon un seuil
    """
    # Normaliser d'abord
    img_norm = (image - tf.reduce_min(image)) / (tf.reduce_max(image) - tf.reduce_min(image) + 1e-8)
    return tf.cast(img_norm > threshold, tf.float32)

def binary_loss(pred, target):
    """
    Loss binaire entre deux images binarisées
    S'assure que pred et target ont la même forme
    """
    # S'assurer que les deux images ont la même taille
    if tf.shape(pred)[1] != tf.shape(target)[1] or tf.shape(pred)[2] != tf.shape(target)[2]:
        pred = tf.image.resize(pred, [tf.shape(target)[1], tf.shape(target)[2]])
    
    # S'assurer que les deux ont le même nombre de canaux
    if pred.shape[-1] != target.shape[-1]:
        if pred.shape[-1] == 1 and target.shape[-1] > 1:
            pred = tf.tile(pred, [1, 1, 1, target.shape[-1]])
        elif target.shape[-1] == 1 and pred.shape[-1] > 1:
            target = tf.tile(target, [1, 1, 1, pred.shape[-1]])
    
    pred_bin = binarize_image(pred)
    target_bin = binarize_image(target)
    
    # Flatten pour BCE
    pred_flat = tf.reshape(pred_bin, [-1])
    target_flat = tf.reshape(target_bin, [-1])
    
    # Loss BCE
    bce = tf.keras.losses.binary_crossentropy(target_flat, pred_flat)
    
    # IoU loss
    intersection = tf.reduce_sum(pred_bin * target_bin)
    union = tf.reduce_sum(pred_bin) + tf.reduce_sum(target_bin) - intersection
    iou_loss = 1.0 - (intersection + 1e-8) / (union + 1e-8)
    
    return tf.reduce_mean(bce) + iou_loss




def reconstruct_with_overlap(patches, num_h, num_w, patch_size, stride, target_h, target_w):
    """
    Reconstruit l'image en moyennant les zones d'overlap
    """
    # Créer canvas vide
    output = tf.zeros([1, target_h, target_w, 1])
    weight_map = tf.zeros([1, target_h, target_w, 1])
    
    # Fonction pour placer un patch
    def place_patch(idx, output, weight_map):
        i = idx // num_w
        j = idx % num_w
        
        y_start = i * stride
        x_start = j * stride
        y_end = tf.minimum(y_start + patch_size[0], target_h)
        x_end = tf.minimum(x_start + patch_size[1], target_w)
        
        # Extraire le patch
        patch = patches[idx]
        
        # Crop si nécessaire
        patch_h = y_end - y_start
        patch_w = x_end - x_start
        patch = patch[:patch_h, :patch_w, :]
        
        # Ajouter au canvas
        patch = tf.expand_dims(patch, 0)
        
        # Créer un masque de padding
        paddings = [
            [0, 0],
            [y_start, target_h - y_end],
            [x_start, target_w - x_end],
            [0, 0]
        ]
        
        patch_padded = tf.pad(patch, paddings, mode='CONSTANT')
        weight_padded = tf.pad(
            tf.ones_like(patch), 
            paddings, 
            mode='CONSTANT'
        )
        
        output = output + patch_padded
        weight_map = weight_map + weight_padded
        
        return idx + 1, output, weight_map
    
    # Boucle sur tous les patches
    total_patches = num_h * num_w
    _, output, weight_map = tf.while_loop(
        lambda idx, *_: idx < total_patches,
        place_patch,
        [tf.constant(0), output, weight_map]
    )
    
    # Moyenne pondérée
    output = output / (weight_map + 1e-8)
    
    return output



@tf.function
def train_step(model, optimizer, vis_image, ir_image, patch_size_vis, overlap=16):
    """
    Train step 
    """
    with tf.GradientTape() as tape:
        vis_gray = tf.reduce_mean(vis_image, axis=-1, keepdims=True)
        ir_gray = tf.reduce_mean(ir_image, axis=-1, keepdims=True) if ir_image.shape[-1] > 1 else ir_image
        
        ir_h = tf.shape(ir_image)[1]
        ir_w = tf.shape(ir_image)[2]
        
        vis_gray = tf.image.resize(vis_gray, size=[ir_h, ir_w], method='bilinear') #introduit une échelle -> plutot
        
        # Normalisation
        vis_gray_norm = (vis_gray - tf.reduce_mean(vis_gray)) / (tf.math.reduce_std(vis_gray) + 1e-8)
        ir_gray_norm = (ir_gray - tf.reduce_mean(ir_gray)) / (tf.math.reduce_std(ir_gray) + 1e-8)
        
        # Stride avec overlap
        stride = patch_size_vis[0] - overlap
        
        # Extraire patches avec overlap
        vis_patches = tf.image.extract_patches(
            images=vis_gray_norm,
            sizes=[1, patch_size_vis[0], patch_size_vis[1], 1],
            strides=[1, stride, stride, 1],  # ← Stride < patch_size
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        
        patches_shape = tf.shape(vis_patches)
        num_patches_h = patches_shape[1]
        num_patches_w = patches_shape[2]
        total_patches = num_patches_h * num_patches_w
        
        vis_patches_batch = tf.reshape(
            vis_patches,
            [total_patches, patch_size_vis[0], patch_size_vis[1], 1]
        )
        
        # Warper les patches
        def warp_patch(patch):
            patch = tf.expand_dims(patch, axis=0)
            flow_pred = model(patch, training=True)
            warped = stn_warp(patch, flow_pred)
            return warped[0], flow_pred[0]
        
        warped_patches_batch, flow_fields_batch = tf.map_fn(
        warp_patch,
        vis_patches_batch,
        fn_output_signature=(tf.float32, tf.float32),
        parallel_iterations=10
        )
        
        # Reconstruction avec moyenne pondérée dans les zones d'overlap
        warped_vis_full = reconstruct_with_overlap(
            warped_patches_batch,
            num_patches_h,
            num_patches_w,
            patch_size_vis,
            stride,
            ir_h,
            ir_w
        )
        
        # LOSSES AMÉLIORÉES
        # 1. NCC Loss - meilleur pour multi-modal
        ncc = ncc_loss(warped_vis_full, ir_gray_norm) #à vérif
        
        # 2. Gradient loss
        grad_loss = gradient_loss(warped_vis_full, ir_gray_norm) #a vérif
        
        # 3. Loss Sobel (gardée mais avec moins de poids)
        sobel_vis = sobel_for_loss_ir(warped_vis_full)
        sobel_ir = sobel_for_loss_ir(ir_image)
        sobel_loss = tf.reduce_mean(tf.abs(sobel_vis - sobel_ir))
        
        # 4. Smoothness regularization sur le flow
        # Reshape flow_fields_batch pour avoir la forme correcte
        flow_fields_4d = tf.reshape(
            flow_fields_batch,
            [num_patches_h, num_patches_w, patch_size_vis[0], patch_size_vis[1], 2]
        )
        smooth_loss = smoothness_loss(flow_fields_4d)
        
        # 5. Binary loss (optionnelle)
        bin_loss = binary_loss(warped_vis_full, ir_image)
        
        # Loss totale avec poids équilibrés
        total_loss = (
            0.4 * ncc +              # Alignement global
            0.3 * grad_loss +        # Alignement des structures
            0.1 * sobel_loss +       # Edges
            0.1 * bin_loss +         # Binarisation
            0.1 * smooth_loss        # Régularisation
        )
        
        # Gradients et update
        grads = tape.gradient(total_loss, model.trainable_variables)
        
        # Gradient clipping pour stabilité
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        return total_loss
    

#rgb -> infra rouge + potentiellement rajouté un mask
#loss binaire / binariser les images
#aller vers la taille du swir modifier pour prendre la même info du côté swir et vis.


@tf.function
def test_step(model,vis_image, ir_image, patch_size_vis):
    """
    Test step amélioré avec toutes les métriques de validation
    """
    # Convertir RGB en grayscale
    vis_gray = tf.reduce_mean(vis_image, axis=-1, keepdims=True)
    
    # S'assurer que IR a 1 canal
    if ir_image.shape[-1] > 1:
        ir_gray = tf.reduce_mean(ir_image, axis=-1, keepdims=True)
    else:
        ir_gray = ir_image
    
    # Resize VIS pour matcher IR
    ir_h = tf.shape(ir_image)[1]
    ir_w = tf.shape(ir_image)[2]
    
    vis_gray = tf.image.resize(
        vis_gray,
        size=[ir_h, ir_w],
        method='bilinear'
    )
    
    # Normaliser comme dans train_step
    vis_gray_norm = (vis_gray - tf.reduce_mean(vis_gray)) / (tf.math.reduce_std(vis_gray) + 1e-8)
    ir_gray_norm = (ir_gray - tf.reduce_mean(ir_gray)) / (tf.math.reduce_std(ir_gray) + 1e-8)
    
    # Extraire les patches VIS
    vis_patches = tf.image.extract_patches(
        images=vis_gray_norm,
        sizes=[1, patch_size_vis[0], patch_size_vis[1], 1],
        strides=[1, patch_size_vis[0], patch_size_vis[1], 1],
        rates=[1, 1, 1, 1],
        padding='VALID'
    )
    
    patches_shape = tf.shape(vis_patches)
    num_patches_h = patches_shape[1]
    num_patches_w = patches_shape[2]
    
    # Reshape
    total_patches = num_patches_h * num_patches_w
    vis_patches_batch = tf.reshape(
        vis_patches,
        [total_patches, patch_size_vis[0], patch_size_vis[1], 1]
    )
    
    # Warper tous les patches (SANS training) et collecter les flows
    def warp_patch(patch):
        patch = tf.expand_dims(patch, axis=0)
        flow_pred = model(patch, training=False)
        warped = stn_warp(patch, flow_pred)
        return warped[0], flow_pred[0]
    
    warped_patches_batch, flow_fields_batch = tf.map_fn(
    warp_patch,
    vis_patches_batch,
    fn_output_signature=(tf.float32, tf.float32),
    parallel_iterations=10
    )
    
    # Reshape en grille 4D
    warped_patches_4d = tf.reshape(
        warped_patches_batch,
        [num_patches_h, num_patches_w, patch_size_vis[0], patch_size_vis[1], 1]
    )
    
    # Permuter
    warped_permuted = tf.transpose(warped_patches_4d, [0, 2, 1, 3, 4])
    
    # Reshape final
    output_h = num_patches_h * patch_size_vis[0]
    output_w = num_patches_w * patch_size_vis[1]
    
    warped_vis_full = tf.reshape(
        warped_permuted,
        [1, output_h, output_w, 1]
    )
    
    # Ajustement final pour matcher IR
    warped_h = tf.shape(warped_vis_full)[1]
    warped_w = tf.shape(warped_vis_full)[2]
    
    if warped_h > ir_h or warped_w > ir_w:
        warped_vis_full = warped_vis_full[:, :ir_h, :ir_w, :]
    elif warped_h < ir_h or warped_w < ir_w:
        pad_h = ir_h - warped_h
        pad_w = ir_w - warped_w
        warped_vis_full = tf.pad(
            warped_vis_full,
            [[0, 0], [0, pad_h], [0, pad_w], [0, 0]],
            mode='REFLECT'
        )
    
    # CALCUL DE TOUTES LES MÉTRIQUES (comme dans train_step)
    
    # 1. NCC Loss
    ncc = ncc_loss(warped_vis_full, ir_gray_norm)
    
    # 2. Gradient loss
    grad_loss = gradient_loss(warped_vis_full, ir_gray_norm)
    
    # 3. Sobel loss
    sobel_vis = sobel_for_loss_ir(warped_vis_full)
    sobel_ir = sobel_for_loss_ir(ir_gray_norm)
    sobel_loss = tf.reduce_mean(tf.abs(sobel_vis - sobel_ir))
    
    # 4. Binary loss
    bin_loss = binary_loss(warped_vis_full, ir_gray_norm)
    
    # 5. Smoothness du flow
    flow_fields_4d = tf.reshape(
        flow_fields_batch,
        [num_patches_h, num_patches_w, patch_size_vis[0], patch_size_vis[1], 2]
    )
    smooth_loss = smoothness_loss(flow_fields_4d)
    
    # Loss totale (même pondération que training)
    total_loss = (
        0.4 * ncc +
        0.3 * grad_loss +
        0.1 * sobel_loss +
        0.1 * bin_loss +
        0.1 * smooth_loss
    )
    
    # MÉTRIQUES SUPPLÉMENTAIRES POUR ANALYSE
    
    # SSIM (Structural Similarity Index)
    ssim_value = tf.image.ssim(warped_vis_full, ir_gray_norm, max_val=2.0)
    
    # MAE (Mean Absolute Error)
    mae = tf.reduce_mean(tf.abs(warped_vis_full - ir_gray_norm))
    
    # MSE (Mean Squared Error)
    mse = tf.reduce_mean(tf.square(warped_vis_full - ir_gray_norm))
    
    # PSNR (Peak Signal-to-Noise Ratio)
    psnr_value = tf.image.psnr(warped_vis_full, ir_gray_norm, max_val=2.0)
    
    # Magnitude moyenne du flow (pour voir si le modèle apprend)
    flow_magnitude = tf.sqrt(
        tf.reduce_sum(flow_fields_batch ** 2, axis=-1)
    )
    mean_flow_magnitude = tf.reduce_mean(flow_magnitude)
    max_flow_magnitude = tf.reduce_max(flow_magnitude)
    
    return {
        # Losses principales
        'total_loss': total_loss,
        'ncc': ncc,
        'gradient': grad_loss,
        'sobel_loss': sobel_loss,
        'bin_loss': bin_loss,
        'smooth_loss': smooth_loss,
        
        # Métriques de qualité
        'ssim': tf.reduce_mean(ssim_value),
        'mae': mae,
        'mse': mse,
        'psnr': tf.reduce_mean(psnr_value),
        
        # Statistiques du flow
        'mean_flow_magnitude': mean_flow_magnitude,
        'max_flow_magnitude': max_flow_magnitude,
        
        # Images pour visualisation
        'warped_vis': warped_vis_full,
        'vis_gray': vis_gray,
        'ir_gray': ir_gray,
        'vis_gray_norm': vis_gray_norm,
        'ir_gray_norm': ir_gray_norm,
        'sobel_vis': sobel_vis,
        'sobel_ir': sobel_ir,
        
        # Flow field moyen (pour visualisation)
        'flow_field': tf.reduce_mean(flow_fields_batch, axis=0)
    }


def visualize_test_results(model,vis_batch, ir_batch, patch_size_ir):
    """
    Fonction complète de visualisation des résultats de test
    """
    # Exécuter le test
    results = test_step(model,vis_batch, ir_batch, patch_size_ir)
    
    # Créer la figure avec plus de subplots
    fig = plt.figure(figsize=(22, 12))
    gs = fig.add_gridspec(3, 5, hspace=0.3, wspace=0.3)
    
    # ==================== LIGNE 1: Images principales ====================
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(results['vis_gray'][0, :, :, 0], cmap='gray')
    ax1.set_title('VIS Original (Gray)', fontsize=11, weight='bold')
    ax1.axis('off')
    
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(results['warped_vis'][0, :, :, 0], cmap='gray')
    ax2.set_title('VIS Warped', fontsize=11, weight='bold')
    ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(results['ir_gray'][0, :, :, 0], cmap='gray')
    ax3.set_title('IR Target', fontsize=11, weight='bold')
    ax3.axis('off')
    
    # Overlay normalisé
    ax4 = fig.add_subplot(gs[0, 3])
    ir_norm = results['ir_gray'][0, :, :, 0].numpy()
    vis_norm = results['warped_vis'][0, :, :, 0].numpy()
    ir_norm = (ir_norm - ir_norm.min()) / (ir_norm.max() - ir_norm.min() + 1e-8)
    vis_norm = (vis_norm - vis_norm.min()) / (vis_norm.max() - vis_norm.min() + 1e-8)
    
    overlay = np.stack([ir_norm, vis_norm, np.zeros_like(ir_norm)], axis=-1)
    ax4.imshow(overlay)
    ax4.set_title('Overlay\n(IR=Red, VIS=Green)', fontsize=11, weight='bold')
    ax4.axis('off')
    
    # Différence absolue
    ax5 = fig.add_subplot(gs[0, 4])
    diff = tf.abs(results['warped_vis'] - results['ir_gray'])
    im5 = ax5.imshow(diff[0, :, :, 0], cmap='hot')
    ax5.set_title(f'Absolute Difference\nMAE: {results["mae"].numpy():.4f}', 
                  fontsize=11, weight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # ==================== LIGNE 2: Gradients et Sobel ====================
    ax6 = fig.add_subplot(gs[1, 0])
    ax6.imshow(results['sobel_vis'][0, :, :, 0], cmap='gray')
    ax6.set_title('Sobel VIS', fontsize=11, weight='bold')
    ax6.axis('off')
    
    ax7 = fig.add_subplot(gs[1, 1])
    ax7.imshow(results['sobel_ir'][0, :, :, 0], cmap='gray')
    ax7.set_title('Sobel IR', fontsize=11, weight='bold')
    ax7.axis('off')
    
    ax8 = fig.add_subplot(gs[1, 2])
    sobel_diff = tf.abs(results['sobel_vis'] - results['sobel_ir'])
    im8 = ax8.imshow(sobel_diff[0, :, :, 0], cmap='hot')
    ax8.set_title(f'Sobel Difference\nLoss: {results["sobel_loss"].numpy():.4f}', 
                  fontsize=11, weight='bold')
    ax8.axis('off')
    plt.colorbar(im8, ax=ax8, fraction=0.046, pad=0.04)
    
    # Visualisation du flow field
    ax9 = fig.add_subplot(gs[1, 3])
    flow = results['flow_field'].numpy()
    # Sous-échantillonner pour la lisibilité
    step = max(1, flow.shape[0] // 20)
    Y, X = np.mgrid[0:flow.shape[0]:step, 0:flow.shape[1]:step]
    U = flow[::step, ::step, 0]
    V = flow[::step, ::step, 1]
    ax9.quiver(X, Y, U, -V, scale=1, scale_units='xy', angles='xy', color='red')
    ax9.set_aspect('equal')
    ax9.set_title(f'Flow Field\nMean: {results["mean_flow_magnitude"].numpy():.2f}px', 
                  fontsize=11, weight='bold')
    ax9.invert_yaxis()
    ax9.grid(True, alpha=0.3)
    
    # Magnitude du flow
    ax10 = fig.add_subplot(gs[1, 4])
    flow_magnitude = np.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)
    im10 = ax10.imshow(flow_magnitude, cmap='jet')
    ax10.set_title(f'Flow Magnitude\nMax: {results["max_flow_magnitude"].numpy():.2f}px', 
                   fontsize=11, weight='bold')
    ax10.axis('off')
    plt.colorbar(im10, ax=ax10, fraction=0.046, pad=0.04)
    
    # ==================== LIGNE 3: Métriques et statistiques ====================
    # Texte des métriques
    ax11 = fig.add_subplot(gs[2, :3])
    ax11.axis('off')
    
    # Graphique en barres des losses
    ax12 = fig.add_subplot(gs[2, 3:])
    losses = {
        'NCC': results['ncc'].numpy(),
        'Gradient': results['gradient'].numpy(),
        'Sobel': results['sobel_loss'].numpy(),
        'Binary': results['bin_loss'].numpy(),
        'Smooth': results['smooth_loss'].numpy()
    }
    
    colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6c5ce7']
    bars = ax12.bar(losses.keys(), losses.values(), color=colors, alpha=0.7, edgecolor='black')
    ax12.set_ylabel('Loss Value', fontsize=11, weight='bold')
    ax12.set_title('Loss Components', fontsize=12, weight='bold')
    ax12.grid(True, alpha=0.3, axis='y')
    ax12.tick_params(axis='x', rotation=45)
    
    # Ajouter les valeurs sur les barres
    for bar in bars:
        height = bar.get_height()
        ax12.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig, results