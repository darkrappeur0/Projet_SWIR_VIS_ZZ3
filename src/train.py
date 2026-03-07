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
    
    flow_x = flow_field[0, :, :, 0] / tf.cast(width, tf.float32) * 2.0
    flow_y = flow_field[0, :, :, 1] / tf.cast(height, tf.float32) * 2.0
    
    x_new = tf.clip_by_value(x_grid + flow_x, -1.0, 1.0)
    y_new = tf.clip_by_value(y_grid + flow_y, -1.0, 1.0)
    
    x_new_pix = (x_new + 1.0) * tf.cast(width - 1, tf.float32) / 2.0
    y_new_pix = (y_new + 1.0) * tf.cast(height - 1, tf.float32) / 2.0
    
    x0 = tf.clip_by_value(tf.cast(tf.floor(x_new_pix), tf.int32), 0, width - 1)
    x1 = tf.clip_by_value(x0 + 1, 0, width - 1)
    y0 = tf.clip_by_value(tf.cast(tf.floor(y_new_pix), tf.int32), 0, height - 1)
    y1 = tf.clip_by_value(y0 + 1, 0, height - 1)
    
    wx = tf.expand_dims(x_new_pix - tf.cast(x0, tf.float32), axis=-1)
    wy = tf.expand_dims(y_new_pix - tf.cast(y0, tf.float32), axis=-1)
    
    def gather_pixel(y_idx, x_idx):
        indices = tf.stack([y_idx, x_idx], axis=-1)
        return tf.gather_nd(image[0], indices)
    
    Q00 = gather_pixel(y0, x0)
    Q01 = gather_pixel(y0, x1)
    Q10 = gather_pixel(y1, x0)
    Q11 = gather_pixel(y1, x1)
    
    top = Q00 * (1.0 - wx) + Q01 * wx
    bottom = Q10 * (1.0 - wx) + Q11 * wx
    interpolated = top * (1.0 - wy) + bottom * wy
    
    return tf.expand_dims(interpolated, 0)


def reconstruct_with_overlap(patches, num_h, num_w, patch_size, stride, target_h, target_w):
    """
    Reconstruit l'image en moyennant les zones d'overlap.
    Utilise tensor_scatter_nd_add pour placer chaque patch directement
    dans le canvas sans tf.pad — évite les paddings négatifs sur les
    patches de bordure issus du padding SAME.

    patches    : Tensor (num_h*num_w, patch_h, patch_w, 1)
    num_h/num_w: int — nombre de patches en hauteur / largeur
    patch_size : tuple (patch_h, patch_w)
    stride     : int — stride utilisé lors de extract_patches
    target_h/w : int — dimensions de l'image de sortie
    """
    # Canvas et weight map à plat sur (H*W, 1) pour scatter
    canvas     = tf.zeros([target_h * target_w, 1])
    weight_map = tf.zeros([target_h * target_w, 1])

    for i in tf.range(num_h):
        for j in tf.range(num_w):
            idx = i * num_w + j

            y_start = i * stride
            x_start = j * stride
            y_end   = tf.minimum(y_start + patch_size[0], target_h)
            x_end   = tf.minimum(x_start + patch_size[1], target_w)

            # Taille réelle de la zone valide dans le canvas
            ph = y_end - y_start   # peut être < patch_size sur les bords
            pw = x_end - x_start

            # Ne garder que la partie valide du patch
            patch_crop = patches[idx, :ph, :pw, :]   # (ph, pw, 1)

            # Construire les indices 1-D (y*W + x) pour chaque pixel du patch
            ys = tf.range(y_start, y_end)             # (ph,)
            xs = tf.range(x_start, x_end)             # (pw,)
            grid_y, grid_x = tf.meshgrid(ys, xs, indexing='ij')   # (ph, pw)
            flat_indices = tf.reshape(grid_y * target_w + grid_x, [-1, 1])  # (ph*pw, 1)

            patch_flat  = tf.reshape(patch_crop, [-1, 1])          # (ph*pw, 1)
            ones_flat   = tf.ones_like(patch_flat)

            canvas     = tf.tensor_scatter_nd_add(canvas,     flat_indices, patch_flat)
            weight_map = tf.tensor_scatter_nd_add(weight_map, flat_indices, ones_flat)

    output = canvas / (weight_map + 1e-8)
    return tf.reshape(output, [1, target_h, target_w, 1])


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
        
        vis_gray = tf.image.resize(vis_gray, size=[ir_h, ir_w], method='bilinear')
        
        # Normalisation
        vis_gray_norm = (vis_gray - tf.reduce_mean(vis_gray)) / (tf.math.reduce_std(vis_gray) + 1e-8)
        ir_gray_norm  = (ir_gray  - tf.reduce_mean(ir_gray))  / (tf.math.reduce_std(ir_gray)  + 1e-8)
        
        # Stride avec overlap
        stride = patch_size_vis[0] - overlap
        
        # Extraire patches avec overlap
        vis_patches = tf.image.extract_patches(
            images=vis_gray_norm,
            sizes=[1, patch_size_vis[0], patch_size_vis[1], 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        
        patches_shape  = tf.shape(vis_patches)
        num_patches_h  = patches_shape[1]
        num_patches_w  = patches_shape[2]
        total_patches  = num_patches_h * num_patches_w
        
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
        
        # Reconstruction corrigée
        warped_vis_full = reconstruct_with_overlap(
            warped_patches_batch,
            num_patches_h,
            num_patches_w,
            patch_size_vis,
            stride,
            ir_h,
            ir_w
        )
        
        # 1. NCC Loss
        ncc = ncc_loss(warped_vis_full, ir_gray_norm)
        
        # 2. Gradient loss
        grad_loss = gradient_loss(warped_vis_full, ir_gray_norm)
        
        # 3. Sobel loss  —  utiliser ir_gray_norm (cohérence avec test_step)
        sobel_vis  = sobel_for_loss_ir(warped_vis_full)
        sobel_ir   = sobel_for_loss_ir(ir_gray_norm)
        sobel_loss = tf.reduce_mean(tf.abs(sobel_vis - sobel_ir))
        
        # 4. Smoothness regularization
        flow_fields_4d = tf.reshape(
            flow_fields_batch,
            [num_patches_h, num_patches_w, patch_size_vis[0], patch_size_vis[1], 2]
        )
        smooth_loss = smoothness_loss(flow_fields_4d)
        
        # 5. Binary loss  —  utiliser ir_gray_norm (cohérence avec test_step)
        bin_loss = binary_loss(warped_vis_full, ir_gray_norm)
        
        # Loss totale
        total_loss = (
            0.4 * ncc        +   # Alignement global
            0.3 * grad_loss  +   # Alignement des structures
            0.1 * sobel_loss +   # Edges
            0.1 * bin_loss   +   # Binarisation
            0.1 * smooth_loss    # Régularisation
        )
        
        grads, _ = tf.clip_by_global_norm(
            tape.gradient(total_loss, model.trainable_variables), 1.0
        )
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        return total_loss
    

#rgb -> infra rouge + potentiellement rajouté un mask
#loss binaire / binariser les images
#aller vers la taille du swir modifier pour prendre la même info du côté swir et vis.


