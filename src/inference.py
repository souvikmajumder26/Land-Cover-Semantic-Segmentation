import os
import sys
import yaml
from yaml import SafeLoader
from pathlib import Path

from constants import Constants
from logger import custom_logger

import cv2
from tensorflow import keras
from keras.models import load_model
from matplotlib import pyplot as plt
import segmentation_models as sm
from data import preprocess_img, postprocess_mask
from smooth_tiled_predictions import predict_img_with_smooth_windowing

if __name__ == "__main__":

    ####################################### Loading Config Values ########################################

    # get the desired parent directory as root path
    ROOT = Path(__file__).resolve().parents[1]

    # add ROOT to sys.path if not present
    if str(ROOT) not in sys.path:
        # add ROOT to sys.path
        sys.path.append(str(ROOT))

    # load the config and parse it into a dictionary
    with open(ROOT / Constants.CONFIG_PATH.value) as f:
        slice_config = yaml.load(f, Loader = SafeLoader)

    # get the required variable values from config
    backbone = slice_config['model']['backbone']        # the backbone/encoder of the model
    patch_size = slice_config['general']['patch_size']  # size of each patch and window
    n_classes = slice_config['general']['n_classes']    # amount of categories to be predicted per pixel
    log_level = slice_config['log']['log_level']

    # get the log file dir from config
    log_dir = ROOT / slice_config['log']['log_dir']
    # make the directory if it does not exist
    log_dir.mkdir(parents = True, exist_ok = True)
    # get the log file path
    log_path = log_dir / slice_config['log']['log_name']
    # convert the path to string in a format compliant with the current OS
    log_path = log_path.as_posix()

    logger = custom_logger("Land Cover Semantic Segmentation Logs", log_path, log_level)

    # get the dir of input images for inference from config
    img_dir = ROOT / slice_config['general']['data_dir'] / slice_config['inference']['data_dir'] / slice_config['inference']['images_dir']
    # check if the model path exists
    if not img_dir.exists():
        logger.error("Images for inference do not exist at %s!" % img_dir)
    img_dir = img_dir.as_posix()

    # get the dir of labelled masks for inference from config
    # NOTE: comment out this part if not providing labelled masks for images to inference on
    labelled_mask_dir = ROOT / slice_config['general']['data_dir'] / slice_config['inference']['data_dir'] / slice_config['inference']['masks_dir']
    # check if the model path exists
    if not labelled_mask_dir.exists():
        logger.error("Masks for inference do not exist at %s!" % labelled_mask_dir)
    labelled_mask_dir = labelled_mask_dir.as_posix()

    # get the predicted masks dir from config
    pred_mask_dir = ROOT / slice_config['general']['data_dir'] / slice_config['inference']['data_dir'] / slice_config['inference']['predicted_masks_dir']
    # make the directory if it does not exist
    pred_mask_dir.mkdir(parents = True, exist_ok = True)
    pred_mask_dir = pred_mask_dir.as_posix()

    # get the prediction plots dir from config
    pred_plot_dir = ROOT / slice_config['general']['data_dir'] / slice_config['inference']['data_dir'] / slice_config['inference']['prediction_plots_dir']
    # make the directory if it does not exist
    pred_plot_dir.mkdir(parents = True, exist_ok = True)
    pred_plot_dir = pred_plot_dir.as_posix()

    # get the model path from config
    model_path = ROOT / slice_config['model']['model_dir'] / slice_config['model']['model_name']
    # check if the model path exists
    if not model_path.exists():
        logger.error("Model does not exist at %s!" % model_path)
    # convert the path to a string in a format compliant with the current OS
    model_path = model_path.as_posix()
    
    ###########################################################################################################

    ####################################### Functional Part of Program ########################################

    preprocess_input = sm.get_preprocessing(backbone)

    for filename in os.listdir(img_dir):

        img = cv2.imread(os.path.join(img_dir, filename), 1)   # read as BGR
        # img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)   # test with this if RAM is too less to process large images
        input_img = preprocess_img(img, backbone)
        input_img = preprocess_input(input_img)

        # NOTE: comment out this part if not providing labelled masks for images to inference on
        labelled_mask = cv2.imread(os.path.join(labelled_mask_dir, filename), 0)   # read as grayscale
        # labelled_mask = cv2.resize(labelled_mask, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)   # test with this if RAM is too less to process large images

        model = load_model(model_path, compile=False)

        # Model prediction using smooth blending
        # Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
        # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
        predictions_smooth = predict_img_with_smooth_windowing(
            input_img,
            window_size=patch_size,
            subdivisions=2,  # Minimal amount of overlap for windowing. Must be an even number
            nb_classes=n_classes,
            pred_func=(
                lambda img_batch_subdiv: model.predict((img_batch_subdiv))
            )
        )

        pred_mask = postprocess_mask(predictions_smooth)

        # Save pred mask with file name and extension same as the input image and mask
        cv2.imwrite(os.path.join(pred_mask_dir, filename), pred_mask)

        # Plot the predictions and save them
        plt.figure(figsize=(18, 4))
        # plotting the input image
        plt.subplot(131)
        plt.title('Test Image')
        plt.imshow(img)
        # plotting the labelled mask
        # NOTE: comment out this part, and change subplot numbering if not providing labelled masks for images to inference on
        plt.subplot(132)
        plt.title('Test Label')
        plt.imshow(labelled_mask)
        # plotting the predicted mask
        plt.subplot(133)
        plt.title('Predicted mask with smooth blending')
        plt.imshow(pred_mask)
        # saving and showing the plot
        plt.savefig(os.path.join(pred_plot_dir, filename.replace('.tif', '.png')))
        plt.show()

        ###########################################################################################################