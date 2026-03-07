from .crop import *
from .loss import *
from .load_data import *
from .neuronnes import *

from .train import stn_warp, reconstruct_with_overlap


@tf.function
def test_step(model, vis_image, ir_image, patch_size_vis, overlap=16):
    """
    Test step avec overlap identique au train_step
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

    # --- Stride avec overlap (identique au train) ---
    stride = patch_size_vis[0] - overlap

    # Extraire les patches VIS avec overlap
    vis_patches = tf.image.extract_patches(
        images=vis_gray_norm,
        sizes=[1, patch_size_vis[0], patch_size_vis[1], 1],
        strides=[1, stride, stride, 1],   # ← stride < patch_size, comme en train
        rates=[1, 1, 1, 1],
        padding='SAME'                    # ← SAME comme en train
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

    # --- Reconstruction avec moyenne pondérée sur les zones d'overlap ---
    warped_vis_full = reconstruct_with_overlap(
        warped_patches_batch,
        num_patches_h,
        num_patches_w,
        patch_size_vis,
        stride,
        ir_h,
        ir_w
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


def visualize_test_results(model, vis_batch, ir_batch, patch_size_ir, overlap=16):
    """
    Fonction complète de visualisation des résultats de test
    """
    # Exécuter le test
    results = test_step(model, vis_batch, ir_batch, patch_size_ir, overlap=overlap)
    
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
    
    for bar in bars:
        height = bar.get_height()
        ax12.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.4f}',
                 ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return fig, results