
"""

Fichiers de configuration pour la méthode u_net

"""
target_shape_img_vis = (2048,1536,3) #height, wide, channels for visible image / A modifié pour les images utilisé
target_shape_img_infra = (1296,1032,1) #height, wide, channels for infra-red image

patch_size_vis = (512, 512)
patch_size_ir  = (432, 432)


path_vis = "datas/temp/rgb/"
path_infra = "datas/temp/swir/"

n_filters = 32
n_classes = 2
batch_size = 1
epochs = 1
learning_rate = 3e-4


