import sys
from model_evaluation import get_rotated_croped_final_image_from_raster, get_whole_prediction
from matplotlib import pyplot as plt, colors
import rasterio
import cv2

def save_prediction(path_raster_before, path_raster_after, resulting_output_mask_path ):

    pre_processed_image = get_rotated_croped_final_image_from_raster(path_raster_before,path_raster_after)
    resulting_mask = get_whole_prediction(pre_processed_image)

    cmap = colors.ListedColormap(['black','#036A14','#27D644'])
    plt.imsave(resulting_output_mask_path,resulting_mask,cmap = cmap)

if __name__ == '__main__':

    raster_before_path = sys.argv[1]
    raster_after_path = sys.argv[2]
    resulting_output_mask_path = sys.argv[3]

    save_prediction(raster_before_path,raster_after_path,resulting_output_mask_path)




