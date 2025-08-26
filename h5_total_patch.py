import h5py
f = h5py.File('./processed_camelyon/patch_feats_pretrain_resnet50_1024.h5')
# Get any slide
slide_id = list(f.keys())[0]
print(f[slide_id]['feat'].shape)  # Should be [num_patches, 1024]

