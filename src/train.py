
from .crop import *
from .loss import *
from .load_data import *
from .neuronnes import *

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


