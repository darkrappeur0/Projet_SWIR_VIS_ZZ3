from .setup_import import *

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
