import nibabel as nib

for path in ["/data/jzk/CT/data_522/test/ccta/1079536.nii.gz", "input.nii.gz"]:
    img = nib.load(path)
    print(path)
    print("shape:", img.shape)
    print("spacing:", img.header.get_zooms()[:3])
    print("orientation:", nib.aff2axcodes(img.affine))
    print("affine:\n", img.affine)