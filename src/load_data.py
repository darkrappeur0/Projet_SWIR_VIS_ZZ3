
from .setup_import import *


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





def image_generator(file_list_vis, file_list_ir, path_vis, path_infra, patch_size_vis, patch_size_ir):
        for vis_name, ir_name in zip(file_list_vis, file_list_ir):
            vis_img = np.array(Image.open(os.path.join(path_vis, vis_name)).convert('RGB'), dtype=np.float32)/255.0
            ir_img  = np.array(Image.open(os.path.join(path_infra, ir_name)).convert('L'), dtype=np.float32)/255.0
            ir_img  = np.expand_dims(ir_img, axis=-1)

            yield ir_img, vis_img