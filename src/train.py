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
    
    # Flow directement en coordonnees normalisees [-1, 1]
    # Un flow de 0.1 = deplacement de ~43px sur 432 — beaucoup plus facile a apprendre
    flow_x = flow_field[0, :, :, 0]
    flow_y = flow_field[0, :, :, 1]

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


def _reconstruct_np(patches_np, num_h, num_w, patch_h, patch_w,
                    stride, target_h, target_w):
    canvas  = np.zeros((target_h, target_w, 1), dtype=np.float32)
    weights = np.zeros((target_h, target_w, 1), dtype=np.float32)
    for i in range(num_h):
        for j in range(num_w):
            idx     = i * num_w + j
            y_start = i * stride
            x_start = j * stride
            y_end   = min(y_start + patch_h, target_h)
            x_end   = min(x_start + patch_w, target_w)
            ph      = y_end - y_start
            pw      = x_end - x_start
            canvas [y_start:y_end, x_start:x_end, :] += patches_np[idx, :ph, :pw, :]
            weights[y_start:y_end, x_start:x_end, :] += 1.0
    return (canvas / (weights + 1e-8))[np.newaxis].astype(np.float32)


def reconstruct_with_overlap(patches, num_h, num_w, patch_size,
                              stride, target_h, target_w):
    result = tf.numpy_function(
        func=lambda p, nh, nw, ph, pw, s, th, tw: _reconstruct_np(
            p, int(nh), int(nw), int(ph), int(pw), int(s), int(th), int(tw)
        ),
        inp=[patches, num_h, num_w,
             patch_size[0], patch_size[1],
             stride, target_h, target_w],
        Tout=tf.float32
    )
    result = tf.ensure_shape(result, [1, None, None, 1])
    return result


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
        
        # Extraire les patches IR correspondants
        ir_patches = tf.image.extract_patches(
            images=ir_gray_norm,
            sizes=[1, patch_size_vis[0], patch_size_vis[1], 1],
            strides=[1, stride, stride, 1],
            rates=[1, 1, 1, 1],
            padding='SAME'
        )
        ir_patches_batch = tf.reshape(
            ir_patches,
            [total_patches, patch_size_vis[0], patch_size_vis[1], 1]
        )

        # Warper les patches — le modele voit VIS + IR pour estimer le flow
        warped_list = []
        flow_list   = []
        for vis_patch, ir_patch in zip(tf.unstack(vis_patches_batch, axis=0),
                                        tf.unstack(ir_patches_batch,  axis=0)):
            vis_in    = tf.expand_dims(vis_patch, axis=0)
            ir_in     = tf.expand_dims(ir_patch,  axis=0)
            flow_pred = model([vis_in, ir_in], training=True)
            warped    = stn_warp(vis_in, flow_pred)
            warped_list.append(warped[0])
            flow_list.append(flow_pred[0])
        warped_patches_batch = tf.stack(warped_list, axis=0)
        flow_fields_batch    = tf.stack(flow_list,   axis=0)
        
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
        # Normaliser bin_loss et sobel_loss pour eviter qu ils explosent
        # bin_loss ~6-7 normalement -> on la ramene dans le meme ordre que les autres
        bin_loss_norm   = bin_loss   / (tf.stop_gradient(bin_loss)   + 1e-8)
        sobel_loss_norm = sobel_loss / (tf.stop_gradient(sobel_loss) + 1e-8)

        total_loss = (
            0.40 * ncc             +   # Alignement global multimodal
            0.30 * grad_loss       +   # Alignement des structures
            0.15 * sobel_loss_norm +   # Edges pixel a pixel (normalise -> toujours ~1.0)
            0.10 * bin_loss_norm   +   # Binarisation pixel a pixel (normalise -> toujours ~1.0)
            0.05 * smooth_loss         # Regularisation flow
        )
        
        grads, _ = tf.clip_by_global_norm(
            tape.gradient(total_loss, model.trainable_variables), 1.0
        )
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        return total_loss