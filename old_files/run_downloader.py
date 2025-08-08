from pipeline.config import PipelineConfig, load_config
from downloader import download_imagery

# Load config with overrides or YAML
config = load_config(bbox=(-74.01, 40.70, -73.99, 40.72))  # Or load_config("config.yaml")
image_path = download_imagery(config, source="Satellite")
print(f"Downloaded GeoTIFF: {image_path}")