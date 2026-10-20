# Satellite-Image-Segmentation-Using-Vision-Transformers

An educational deep-learning project for pixel-wise segmentation of satellite and drone imagery. It fine-tunes SegFormer, a transformer-based semantic segmentation model, to label visible regions such as buildings, roads, water, agricultural land, forest, industrial sites, fire, smoke, and floodwater.

Given an overhead RGB image, the model predicts a class for every pixel and creates a color-coded segmentation map. A Google Colab upload workflow lets you select an image, view the prediction and overlay, and download the generated outputs. The companion Python modules separate dataset loading, model setup, training, evaluation, tiled prediction, and visualization.

> **Important:** This project is a learning prototype. It needs paired images and pixel-level masks for training. Adding a class name to the configuration does not teach the model to recognize that class.

## Project goals

This project explores how vision transformers can help automate the interpretation of high-altitude imagery. It demonstrates a complete semantic segmentation workflow:

1. Prepare aligned image and mask pairs.
2. Encode each target region with a class ID.
3. Fine-tune a pretrained SegFormer model on the project's classes.
4. Measure performance on a validation set.
5. Run the trained model on new imagery and inspect its pixel-wise predictions.

The model identifies the appearance of labeled categories. It does not reason about objects hidden from view or guarantee that a prediction is correct in a new location or sensor condition.

## What the project demonstrates

- Load matching image and mask files from a folder-based dataset.
- Apply spatial augmentation jointly to images and masks to preserve alignment.
- Fine-tune SegFormer with a custom output head for the selected class set.
- Track training loss, per-class intersection over union (IoU), mean IoU, and pixel accuracy.
- Predict on large images using overlapping tiles while retaining the original image dimensions.
- Visualize predictions as a colored mask and as an overlay on the source image.
- Upload one or more images from Google Colab and download the resulting outputs.

## Model and training approach

The workflow starts from the Hugging Face checkpoint `nvidia/segformer-b0-finetuned-ade-512-512`. Its classification head is replaced to match the project's class count, while compatible pretrained weights are loaded. The model is then fine-tuned on the provided image/mask pairs.

The training defaults are defined in the notebook and in `satellite_segmentation_config.py`:

| Setting | Default |
|---|---:|
| Model checkpoint | `nvidia/segformer-b0-finetuned-ade-512-512` |
| Input tile size | 512 × 512 pixels |
| Batch size | 4 |
| Epochs | 10 |
| Learning rate | 0.00006 |
| Optimizer | AdamW |
| Weight decay | 0.01 |

These are starting values for an educational run. They are not tuned for every dataset, GPU, class balance, or image resolution. Adjust them after reviewing validation performance and available memory.

## Class labels

Each mask is a single-channel PNG. Every pixel stores one integer class ID. The class order must be identical in the notebook, configuration, and all masks.

| ID | Class | Example label meaning |
|---:|---|---|
| 0 | Background | Background or scene pixels assigned to the background class |
| 1 | Building | Visible buildings and rooftops |
| 2 | Road | Visible roads and paved transport routes |
| 3 | Water | Ordinary rivers, lakes, ponds, or other non-flood water |
| 4 | Barren | Bare soil, rock, or other barren ground |
| 5 | Forest | Forested or dense tree-cover regions |
| 6 | Agricultural | Cropland and agricultural fields |
| 7 | Fire | Visible active fire pixels, if resolvable in the imagery |
| 8 | Smoke | Visible smoke plume pixels |
| 9 | Other | Visible category not represented by another class |
| 10 | Occluded/unknown | Ground hidden by smoke, fog, cloud, or another obstruction |
| 11 | Industrial/factory | Factories and industrial sites |
| 12 | Flood | Floodwater, distinct from ordinary water |
| 255 | Ignore/unlabeled | Pixel excluded from training and evaluation loss |

A single mask contains all classes for its paired image; do not create one mask file per class. Ordinary water should use ID 3 and floodwater ID 12 if the model is expected to distinguish them. Use ID 10 when the underlying ground cannot be identified. If a flooded building is visible, choose a consistent annotation policy: label visible roof pixels as building and water-covered pixels as flood, or define and train a separate flooded-building class for that use case.

## Dataset layout and naming

Image and mask files are paired by filename stem. Their extensions can differ, but their names before the extension must match.

```text
data/
  train/
    images/
      tile_001.jpg
      tile_002.jpg
    masks/
      tile_001.png
      tile_002.png
  val/
    images/
      tile_101.jpg
    masks/
      tile_101.png
```

For example, `tile_001.jpg` is paired with `tile_001.png`. Images may be PNG, JPEG, or TIFF. Masks must be PNGs containing integer class IDs from 0 through 12, plus 255 for ignored pixels. RGB color masks must be converted to class-ID masks before training; `mask_encoding.py` includes a helper for exact palette-color conversion.

Masks must align spatially with the images they label. The notebook resizes both to the configured model input size during training. For robust evaluation, split data by geographic area, flight, or acquisition date when possible, rather than randomly splitting neighboring crops from the same scene. Nearby tiles can be nearly identical and may otherwise make validation scores overly optimistic.

Configure `DATA_ROOT` in the notebook's **Imports and configuration** cell or in `satellite_segmentation_config.py`.

## Annotation and data quality

The model learns from the examples in the training masks. Include representative examples of every class you want it to predict, and make sure the validation set also contains examples of those classes. Rare categories such as fire, smoke, and flood often need extra annotation attention because they occupy a small fraction of image pixels.

For smoke recognition, include both smoke and visually similar fog/cloud examples. For flood mapping, include floodwater as well as ordinary rivers, lakes, wet soil, and shadows. For industrial/factory recognition, include different roof materials, factory sizes, and surrounding land uses. Use consistent pixel boundaries and class priorities across annotators.

The project uses RGB images only; it does not ingest thermal or multispectral bands. A single image cannot reveal a building, road, or vegetation hidden by dense smoke, fog, cloud, or water. Mark hidden pixels as occluded/unknown instead of asking the model to infer what is underneath. Small fires, thin smoke plumes, narrow roads, and other small objects may be too small to resolve at high altitude or after image resizing.

## Project files

```text
Satellite_Image_Segmentation_Using_Vision_Transformers.ipynb  # End-to-end Colab notebook
satellite_segmentation_config.py                              # Paths, class list, and training settings
dataset_loader.py                                             # Paired image/mask loading
mask_encoding.py                                              # Class-ID validation and palette conversion
segformer_model.py                                            # SegFormer construction
train_segformer.py                                            # Training entry point
evaluate_segmentation.py                                      # Validation evaluation
segmentation_metrics.py                                       # Confusion matrix and IoU metrics
tile_large_satellite_image.py                                 # Overlapping image tile iterator
predict_satellite_image.py                                   # Checkpoint loading and tiled prediction
upload_satellite_image_colab.py                               # Google Colab upload interface
visualize_segmentation_results.py                             # Color masks and overlays
save_load_segmentation_model.py                               # Checkpoint utilities
satellite_environment_setup.py                                # Dependency install command helper
```

Keep the Python modules in the same directory so their imports resolve. The notebook includes a self-contained version of the workflow and is the recommended entry point for a first run.

## Run in Google Colab

1. Upload `Satellite_Image_Segmentation_Using_Vision_Transformers.ipynb` to Google Colab and open it.
2. Choose **Runtime → Change runtime type** and select a GPU if available.
3. Make the dataset folders available to the runtime. For a Drive dataset, mount Google Drive and set `DATA_ROOT` to the correct folder.
4. Run notebook cells from top to bottom. The dataset check reports missing folders or unmatched filenames.
5. Review the sample image and mask, and confirm that the mask contains the expected class IDs.
6. Run the model initialization and training cells. The best validation checkpoint is saved under `outputs/best_segformer`.
7. After training completes, run **“Upload and classify your own drone or satellite image.”** Select one or more RGB images in the file picker.
8. Review the input, predicted class map, and overlay. Download the segmentation and overlay files before the Colab runtime ends.

Training from scratch on a new dataset requires labeled masks. If the runtime is restarted after training, load the saved checkpoint and preserve the same class order before performing inference. A GPU is recommended; CPU training may take substantially longer.

## Run the Python modules

Create a Python environment and install the core dependencies:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
source .venv/bin/activate
python -m pip install torch torchvision transformers accelerate albumentations opencv-python-headless pillow matplotlib tqdm
```

Set paths, classes, and training parameters in `satellite_segmentation_config.py`. Place the dataset in the documented folder layout, then run training from the project directory:

```bash
python train_segformer.py
```

The training script saves the best checkpoint under `outputs/best_segformer` and stores the class order in `outputs/class_names.json`. The upload interface in `upload_satellite_image_colab.py` is intended for Google Colab. For Python inference, call `predict_array` or `predict_file` from `predict_satellite_image.py`; large images are processed in overlapping tiles.

## Outputs

- `outputs/best_segformer/` — SegFormer checkpoint with the best validation mean IoU seen during training.
- `outputs/class_names.json` — ordered class names used for that checkpoint.
- `<image>_segmentation.png` — color-coded class prediction at the input image dimensions.
- `<image>_overlay.png` — segmentation blended with the input image.
- Training loss, validation IoU, pixel accuracy, and sample visualizations — displayed by the notebook.

Output paths are relative to the notebook or script's current working directory. Colab storage is temporary; download checkpoints and results you want to keep before the runtime is deleted.

## Evaluation

The validation workflow computes a confusion matrix and reports per-class IoU, mean IoU, and pixel accuracy. IoU measures overlap between predicted and ground-truth pixels for a class. Mean IoU summarizes performance across the configured classes, while pixel accuracy reports the fraction of correctly labeled pixels.

Metrics are meaningful only when masks are accurate and the validation scenes represent the intended use. Pixel accuracy can look high even when rare classes are missed because background or common classes dominate the pixel count. Review per-class IoU and visual results, especially for fire, smoke, flood, and industrial sites. Keep a separate test set for final reporting rather than tuning on the validation set.

## Troubleshooting

- **Missing folder error:** verify `DATA_ROOT` and that `train/images`, `train/masks`, `val/images`, and `val/masks` exist.
- **No paired files:** image and mask filenames must share the same stem, for example `tile_001.jpg` and `tile_001.png`.
- **Unexpected mask values:** masks must contain only configured IDs 0–12 or ignore ID 255. Remap source dataset labels before training.
- **RGB mask error:** convert the color-coded mask to a single-channel integer class-ID PNG.
- **Class absent from predictions:** confirm that the class appears in training masks with the correct ID and that enough representative examples are present.
- **CUDA out-of-memory:** lower `BATCH_SIZE` or `IMAGE_SIZE`, or use a GPU with more memory.
- **Upload cell import error:** the `google.colab.files` upload interface is Colab-specific. Use a local file path with `predict_file` outside Colab.
- **Confusing smoke/fog or water/flood predictions:** add accurately labeled examples of both categories and test on scenes from the same sensor and conditions expected at deployment.

## Limitations and responsible use

Adding a class name to the configuration does not teach the model that class. It must be trained with masks containing examples of every desired category. Performance can be unreliable on locations, altitudes, seasons, sensors, weather, or image resolutions that differ from the training data. Smoke can resemble fog or clouds; floodwater can resemble ordinary water or wet ground. Concealed ground cannot be classified from a single RGB frame, and the model does not use thermal information.

The flood class is a visual segmentation label, not by itself a before/after flood-change analysis. Reliable flood extent or damage assessment may require pre-event imagery, geospatial alignment, repeated observations, and expert validation. This educational prototype is not a substitute for expert review in emergency response or other high-stakes decisions.

## Possible extensions

- Build or use a carefully annotated dataset spanning multiple sensors, regions, altitudes, and seasons.
- Add pre-event imagery and change detection for flood extent and damage mapping.
- Add temporal drone/video inputs to use motion and context when distinguishing smoke from fog or clouds.
- Support thermal or multispectral imagery with a model and training data designed for those input bands.
- Address class imbalance with sampling or weighted losses and report per-class metrics on a held-out test set.
- Add confidence maps and flag uncertain regions for human review.
- Export georeferenced predictions for GIS workflows and retain spatial metadata.
- Compare SegFormer against CNN-based segmentation baselines on the same splits.

## Main technologies

- Python and PyTorch
- Hugging Face Transformers (SegFormer)
- Albumentations
- NumPy and Pillow
- Matplotlib and tqdm
- Google Colab for interactive upload and GPU execution

## References

- [SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers](https://arxiv.org/abs/2105.15203)
- [Hugging Face Transformers SegFormer documentation](https://huggingface.co/docs/transformers/model_doc/segformer)
- [PyTorch documentation](https://pytorch.org/docs/)

## Author

PRITHIYADARSHAN T  
Computer Science and Engineering (Artificial Intelligence & Machine Learning)  
GitHub: [github.com/prithyadarshan](https://github.com/prithyadarshan)  
LinkedIn: [linkedin.com/in/prithiyadarshan-thiyagarajan-379a4b2a3](https://www.linkedin.com/in/prithiyadarshan-thiyagarajan-379a4b2a3)
