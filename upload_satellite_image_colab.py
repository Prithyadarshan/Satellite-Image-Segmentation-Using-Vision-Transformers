"""Run this in Google Colab after training to upload and classify imagery."""
from google.colab import files
from IPython.display import display
from PIL import Image
import io
import matplotlib.pyplot as plt
from predict_satellite_image import predict_array
from visualize_segmentation_results import colorize, save_visuals
from satellite_segmentation_config import CLASS_NAMES, CLASS_COLORS

uploaded=files.upload()
for filename,data in uploaded.items():
    image=Image.open(io.BytesIO(data)).convert("RGB")
    mask=predict_array(image)
    color,overlay=save_visuals(image,mask,filename.rsplit('.',1)[0])
    fig,axes=plt.subplots(1,3,figsize=(18,6))
    axes[0].imshow(image); axes[0].set_title("Uploaded image")
    axes[1].imshow(color); axes[1].set_title("Class prediction")
    axes[2].imshow(overlay); axes[2].set_title("Overlay")
    for ax in axes: ax.axis("off")
    plt.show()
