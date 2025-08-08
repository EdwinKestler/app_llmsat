import os
from samgeo import SamGeo

def run_samgeo_on_tif(tif_path, checkpoint_path="checkpoints/sam_vit_h_4b8939.pth", outdir="output/"):
    os.makedirs(outdir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(tif_path))[0]
    mask_tif = os.path.join(outdir, f"{basename}_mask.tif")
    vector_gpkg = os.path.join(outdir, f"{basename}_mask.gpkg")

    sam = SamGeo(
        model_type="vit_h",
        checkpoint=checkpoint_path,
        sam_kwargs=None,
    )

    sam.generate(tif_path, output=mask_tif, foreground=True, unique=True)
    sam.tiff_to_vector(mask_tif, vector_gpkg)

    return mask_tif, vector_gpkg

# Example usage
