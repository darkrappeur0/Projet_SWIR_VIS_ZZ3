from .setup_import import *

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

