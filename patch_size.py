# import h5py

# with h5py.File('./processed_camelyon_old/patches/test_008.h5', 'r') as f:
#     print(list(f.keys()))
#     # You might see 'coords', 'imgs', etc.
#     coords = f['coords'][:] 


import openslide

slide1 = openslide.OpenSlide('/vol/research/scratch1/NOBACKUP/rk01337/BRACS/histoimage.na.icar.cnr.it/BRACS_WSI/train/Group_BT/Type_N/BRACS_1003714.svs')
slide2 = openslide.OpenSlide('/vol/research/datasets/pathology/Camelyon/Camelyon16/training/normal/normal_045.tif')
slide = openslide.OpenSlide('/vol/research/datasets/pathology/tcga/tcga-brca/WSIs/TCGA-A7-A6VV-01Z-00-DX1.07AE0E16-A883-4C86-BC74-4E13081175F2.svs')
print("Level downsamples:", slide.level_downsamples)
print("Level downsamples:", slide.level_downsamples)
print("Level dimensions:", slide.level_dimensions)

