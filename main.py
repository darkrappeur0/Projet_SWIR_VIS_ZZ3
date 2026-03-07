
from config import *
from src.test import *
from src.train import *

visible_list, infra_list = LoadData(path_vis, path_infra)

# Découpe les images en patches
patch_size_vis = (512, 512)
patch_size_ir  = (432, 432)


model = UNetCompiled(
    input_size=(432, 432, 1),  # patch size
    n_filters=n_filters,
    n_classes=n_classes
)

model.summary()

mse_loss = MeanSquaredError()
optimizer = Adam(learning_rate=learning_rate)


vis_train, vis_test, ir_train, ir_test = train_test_split(
    visible_list, infra_list, test_size=0.2, random_state=42, shuffle=True
)


train_ds = tf.data.Dataset.from_generator(
    lambda: image_generator(vis_train, ir_train, path_vis, path_infra, patch_size_vis, patch_size_ir),
    output_signature=(
        tf.TensorSpec(shape=(None,None,1), dtype=tf.float32),   # IR full image
        tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),   # Vis full image
        
    )
)
train_ds = train_ds.batch(1)

# Dataset test
test_ds = tf.data.Dataset.from_generator(
    lambda: image_generator(vis_test, ir_test, path_vis, path_infra, patch_size_vis, patch_size_ir),
    output_signature=(
        tf.TensorSpec(shape=(None,None,1), dtype=tf.float32),
        tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
        
    )
)
test_ds = test_ds.batch(1)


checkpoint = ModelCheckpoint("unet_stn_deformation.h5",save_best_only=True,monitor='loss')
early_stop = EarlyStopping(monitor='loss',patience=10)


import time

# Entraînement avec feature matching
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    epoch_loss = 0
    valid_pairs = 0
    skipped_pairs = 0
    epoch_start = time.time()
    is_valid = True
    
    for step, (ir_batch, vis_batch) in enumerate(train_ds):
        step_start = time.time()
        
        # Utiliser le wrapper qui gère le feature matching
        """
        loss, is_valid = train_step_wrapper(
            vis_batch, ir_batch, patch_size_ir, 
            use_mask=False, 
            debug=(step % 50 == 0)  # Debug tous les 50 steps
        )
        """
        loss = train_step(model,optimizer,vis_batch,ir_batch,patch_size_ir,overlap=16)
        
        if is_valid:
            epoch_loss += loss.numpy()
            valid_pairs += 1
        else:
            skipped_pairs += 1
        
        if step % 2 == 0:
            step_time = time.time() - step_start
            print(f"Step {step}: Loss = {loss.numpy():.4f}, Time = {step_time:.2f}s, Valid pairs: {valid_pairs}, Skipped: {skipped_pairs}")
    
    epoch_time = time.time() - epoch_start
    if valid_pairs > 0:
        avg_loss = epoch_loss / valid_pairs
        print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s")
        print(f"Avg Loss: {avg_loss:.4f}")
        print(f"Valid pairs: {valid_pairs}/{valid_pairs + skipped_pairs} ({valid_pairs/(valid_pairs + skipped_pairs)*100:.1f}%)")
    else:
        print(f"\nEpoch {epoch+1}: No valid pairs found!")



print("🔍 Analyse des résultats de test...\n")

for i, (ir_batch, vis_batch) in enumerate(test_ds.take(3)):  # Tester sur 3 exemples
    
    print(f"\n{'='*80}")
    print(f"  TEST EXEMPLE #{i+1}")
    print(f"{'='*80}\n")
    
    fig, results = visualize_test_results(model,vis_batch, ir_batch, patch_size_ir)
    plt.savefig(f'temp/test_result_{i+1}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Affichage console simplifié
    print(f"✓ Total Loss: {results['total_loss'].numpy():.6f}")
    print(f"✓ SSIM: {results['ssim'].numpy():.4f} | PSNR: {results['psnr'].numpy():.2f} dB")
    print(f"✓ Flow magnitude: {results['mean_flow_magnitude'].numpy():.2f} ± "
          f"{results['max_flow_magnitude'].numpy():.2f} pixels")

 

#mettre un patch qui respecte le ratio de l'image 