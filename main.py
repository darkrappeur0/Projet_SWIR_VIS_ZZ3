from config import *
from src.test import *
from src.train import *

visible_list, infra_list = LoadData(path_vis, path_infra)

# Patch sizes comme tuples Python — requis par tf.image.extract_patches
# (sizes doit etre une constante Python, pas un tenseur)
patch_size_vis = (512, 512)
patch_size_ir  = (432, 432)


model = UNetCompiled(
    input_size=(432, 432, 1),  # taille d un patch (H, W, 1) — x2 en entree (VIS+IR)
    n_filters=n_filters,
    n_classes=n_classes
)

model.summary()

mse_loss  = MeanSquaredError()
optimizer = Adam(learning_rate=learning_rate)


vis_train, vis_test, ir_train, ir_test = train_test_split(
    visible_list, infra_list, test_size=0.2, random_state=42, shuffle=True
)


train_ds = tf.data.Dataset.from_generator(
    lambda: image_generator(vis_train, ir_train, path_vis, path_infra,
                            patch_size_vis, patch_size_ir),
    output_signature=(
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),  # IR
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),  # VIS
    )
)
train_ds = train_ds.batch(1)

test_ds = tf.data.Dataset.from_generator(
    lambda: image_generator(vis_test, ir_test, path_vis, path_infra,
                            patch_size_vis, patch_size_ir),
    output_signature=(
        tf.TensorSpec(shape=(None, None, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
    )
)
test_ds = test_ds.batch(1)


checkpoint = ModelCheckpoint("unet_stn_deformation.h5", save_best_only=True, monitor='loss')
early_stop = EarlyStopping(monitor='loss', patience=10)

import time

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    epoch_loss   = 0
    valid_pairs  = 0
    skipped_pairs = 0
    epoch_start  = time.time()

    for step, (ir_batch, vis_batch) in enumerate(train_ds):
        step_start = time.time()

        loss = train_step(model, optimizer, vis_batch, ir_batch,
                          patch_size_ir, overlap=16)

        epoch_loss  += loss.numpy()
        valid_pairs += 1

        if step % 2 == 0:
            step_time = time.time() - step_start
            print(f"Step {step}: Loss = {loss.numpy():.4f}, "
                  f"Time = {step_time:.2f}s, Valid pairs: {valid_pairs}")

    epoch_time = time.time() - epoch_start
    avg_loss   = epoch_loss / max(valid_pairs, 1)
    print(f"\nEpoch {epoch+1} completed in {epoch_time:.2f}s — Avg Loss: {avg_loss:.4f}")

    # Sauvegarder le modele apres chaque epoch
    model.save("unet_stn_deformation.h5")


print("\nAnalyse des resultats de test...\n")

for i, (ir_batch, vis_batch) in enumerate(test_ds.take(3)):

    print(f"\n{'='*80}")
    print(f"  TEST EXEMPLE #{i+1}")
    print(f"{'='*80}\n")

    fig, results = visualize_test_results(model, vis_batch, ir_batch, patch_size_ir)
    plt.savefig(f'temp/test_result_{i+1}.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"Total Loss : {results['total_loss'].numpy():.6f}")
    print(f"SSIM       : {results['ssim'].numpy():.4f}  |  "
          f"PSNR: {results['psnr'].numpy():.2f} dB")
    print(f"Flow mag   : mean={results['mean_flow_magnitude'].numpy():.2f}px  "
          f"max={results['max_flow_magnitude'].numpy():.2f}px")
 

#mettre un patch qui respecte le ratio de l'image 