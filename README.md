# 🛰️ Proyecto de Segmentación Visual de Imágenes Satelitales

Este programa automatiza el análisis de imágenes satelitales para identificar y mapear elementos visibles en un área específica, con soporte para GPU (CUDA) para acelerar la segmentación.

## 📌 ¿Qué hace este programa?

1. **Descarga automática de imágenes satelitales** usando Esri Satellite.
2. **Segmentación semántica (LangSAM)** con prompts como "árboles", "agua".
3. **Segmentación general (SAM2)** para detectar objetos destacados.
4. **Vectorización y análisis** de áreas (m²) y porcentajes.
5. **Visualización** en mapas interactivos (HTML) y gráficos (Altair).
6. **Exporta** GeoTIFFs, GeoPackages, CSVs, y mapas HTML.

## 🗂️ Archivos generados

- `output/s2harm_rgb_saa.tif`: Imagen satelital.
- `output/segment_<clase>_<fecha>.gpkg`: Segmentos por tipo.
- `output/sam2_mask.tif`: Máscara SAM2.
- `output/segments.gpkg`, `summary.csv`: Resumen vectorial/áreas.
- `output/segmentacion_mapa_<fecha>.html`: Mapa interactivo.

## ✅ Requisitos

- Python 3.9+
- Instalar dependencias: `pip install -r requirements.txt` o `conda env create -f environment.yml`
- **GPU Support**:
  - NVIDIA GPU con CUDA (verifique con `nvidia-smi`).
  - Instalar PyTorch con CUDA: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` (ajuste según su versión de CUDA).
  - Descargar el checkpoint SAM2 (`sam2_hiera_l.pt`) a `checkpoints/` desde [SAM2 repo](https://github.com/facebookresearch/segment-anything-2).
- Configurar `OPENAI_API_KEY` en un archivo `.env` para consultas en lenguaje natural.

## 🚀 Uso

- **CLI**: `python cli.py --bbox -74.01 40.70 -73.99 40.72 --prompt trees --prompt water --device cuda`
- **Streamlit**: `streamlit run app.py` (seleccione `cuda` o `cpu` en la barra lateral).
- Nota: `samgeo_utils.py` está obsoleto (en `old_files/`) y reemplazado por el pipeline modular en `pipeline/`.

## 👨‍💻 Autor

Desarrollado por Edwin Kestler para agricultura de precisión, supervisión urbana y análisis ambiental.