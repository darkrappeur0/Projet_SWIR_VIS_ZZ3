from .crop import *
from .loss import *
from .load_data import *
from .neuronnes import *

@tf.function
def stn_warp(image, flow_field):
    """
    Warping bilinéaire classique.
    Les pixels hors-champ après déplacement sont clampés au bord (STN standard).
    """
    height = tf.shape(image)[1]
    width  = tf.shape(image)[2]

    y_coords = tf.linspace(-1.0, 1.0, height)
    x_coords = tf.linspace(-1.0, 1.0, width)
    x_grid, y_grid = tf.meshgrid(x_coords, y_coords)

    # Flow limité à ±0.3 (~130px sur 432) — suffisant pour du recalage
    flow_x = tf.clip_by_value(flow_field[0, :, :, 0], -0.3, 0.3)
    flow_y = tf.clip_by_value(flow_field[0, :, :, 1], -0.3, 0.3)

    x_new_pix = tf.clip_by_value(
        (tf.clip_by_value(x_grid + flow_x, -1.0, 1.0) + 1.0) * tf.cast(width  - 1, tf.float32) / 2.0,
        0.0, tf.cast(width  - 1, tf.float32)
    )
    y_new_pix = tf.clip_by_value(
        (tf.clip_by_value(y_grid + flow_y, -1.0, 1.0) + 1.0) * tf.cast(height - 1, tf.float32) / 2.0,
        0.0, tf.cast(height - 1, tf.float32)
    )

    x0 = tf.clip_by_value(tf.cast(tf.floor(x_new_pix), tf.int32), 0, width  - 1)
    x1 = tf.clip_by_value(x0 + 1,                                   0, width  - 1)
    y0 = tf.clip_by_value(tf.cast(tf.floor(y_new_pix), tf.int32), 0, height - 1)
    y1 = tf.clip_by_value(y0 + 1,                                   0, height - 1)

    wx = tf.expand_dims(x_new_pix - tf.cast(x0, tf.float32), axis=-1)
    wy = tf.expand_dims(y_new_pix - tf.cast(y0, tf.float32), axis=-1)

    def gather_pixel(y_idx, x_idx):
        return tf.gather_nd(image[0], tf.stack([y_idx, x_idx], axis=-1))

    interpolated = (
        gather_pixel(y0, x0) * (1.0 - wx) * (1.0 - wy) +
        gather_pixel(y0, x1) *        wx  * (1.0 - wy) +
        gather_pixel(y1, x0) * (1.0 - wx) *        wy  +
        gather_pixel(y1, x1) *        wx  *        wy
    )

    return tf.expand_dims(interpolated, 0)


def extract_patches_valid(image_norm, patch_h, patch_w, stride):
    """
    Extrait les patches en mode VALID : aucun zero-padding.
    Tous les pixels des patches proviennent de l'image réelle.
    C'est la correction principale contre les bandes/cadres gris.
    """
    patches = tf.image.extract_patches(
        images=image_norm,
        sizes=[1, patch_h, patch_w, 1],
        strides=[1, stride, stride, 1],
        rates=[1, 1, 1, 1],
        padding='VALID'  # <-- pas de zero-padding
    )
    patches_shape = tf.shape(patches)
    num_h = patches_shape[1]
    num_w = patches_shape[2]
    total = num_h * num_w
    patches_batch = tf.reshape(patches, [total, patch_h, patch_w, 1])
    return patches_batch, num_h, num_w


def train_step(model, optimizer, vis_image, ir_image, patch_size_vis, overlap=16):
    """
    Train step.

    Corrections vs versions précédentes :
      1. padding='VALID' dans extract_patches  →  plus de zero-padding sur les bords
         => plus de bandes/cadres gris dans l'image warpée.
      2. binary_loss retirée (valeur ~5-6, dominait toute la loss).
      3. Poids : NCC 55 %, gradient 35 %, smoothness 10 %.
    """
    with tf.GradientTape() as tape:

        vis_gray = tf.reduce_mean(vis_image, axis=-1, keepdims=True)
        ir_gray  = (tf.reduce_mean(ir_image, axis=-1, keepdims=True)
                    if ir_image.shape[-1] > 1 else ir_image)

        ir_h = tf.shape(ir_image)[1]
        ir_w = tf.shape(ir_image)[2]

        vis_gray = tf.image.resize(vis_gray, [ir_h, ir_w], method='bilinear')

        vis_gray_norm = (vis_gray - tf.reduce_mean(vis_gray)) / (tf.math.reduce_std(vis_gray) + 1e-8)
        ir_gray_norm  = (ir_gray  - tf.reduce_mean(ir_gray))  / (tf.math.reduce_std(ir_gray)  + 1e-8)

        stride  = patch_size_vis[0] - overlap
        patch_h = patch_size_vis[0]
        patch_w = patch_size_vis[1]

        vis_patches_batch, num_patches_h, num_patches_w = extract_patches_valid(
            vis_gray_norm, patch_h, patch_w, stride
        )
        ir_patches_batch, _, _ = extract_patches_valid(
            ir_gray_norm, patch_h, patch_w, stride
        )

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

        ncc_list    = []
        grad_list   = []
        smooth_list = []

        for warped_patch, ir_patch, flow_patch in zip(
            tf.unstack(warped_patches_batch, axis=0),
            tf.unstack(ir_patches_batch,     axis=0),
            tf.unstack(flow_fields_batch,    axis=0)
        ):
            w = tf.expand_dims(warped_patch, 0)
            t = tf.expand_dims(ir_patch,     0)
            f = tf.expand_dims(flow_patch,   0)

            ncc_list.append(ncc_loss(w, t))
            grad_list.append(gradient_loss(w, t))
            smooth_list.append(smoothness_loss(f))

        ncc         = tf.reduce_mean(tf.stack(ncc_list))
        grad_loss_v = tf.reduce_mean(tf.stack(grad_list))
        smooth_loss = tf.reduce_mean(tf.stack(smooth_list))

        total_loss = (
            0.55 * ncc         +
            0.35 * grad_loss_v +
            0.10 * smooth_loss 
        )

        grads, _ = tf.clip_by_global_norm(
            tape.gradient(total_loss, model.trainable_variables), 1.0
        )
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return total_loss