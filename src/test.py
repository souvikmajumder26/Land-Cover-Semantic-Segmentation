import os
import sys
import yaml
from yaml import SafeLoader
from pathlib import Path

from utils.constants import Constants
from utils.logger import custom_logger

import cv2
from tensorflow import keras
from keras.models import load_model

from utils.data_processing import preprocess_img, postprocess_mask
from utils.smooth_tiled_predictions import predict_img_with_smooth_windowing
from utils.plots import plot_test


def get_root():
    # get the desired parent directory as root path
    ROOT = Path(__file__).resolve().parents[1]

    # add ROOT to sys.path if not present
    if str(ROOT) not in sys.path:
        # add ROOT to sys.path
        sys.path.append(str(ROOT))

    # load the config and parse it into a dictionary
    with open(ROOT / Constants.CONFIG_PATH.value) as f:
        slice_config = yaml.load(f, Loader = SafeLoader)
    
    return ROOT, slice_config


if __name__ == "__main__":

    ################################# Loading Variables and Paths from Config #################################

    ROOT, slice_config = get_root()

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
    
    # initialize the logger
    logger = custom_logger("Land Cover Semantic Segmentation Testing Logs", log_path, log_level)

    # get the dir of input images for inference from config
    img_dir = ROOT / slice_config['general']['data_dir'] / slice_config['inference']['data_dir'] / slice_config['inference']['images_dir']
    img_dir = img_dir.as_posix()

    # get the dir of labelled masks for inference from config
    labelled_mask_dir = ROOT / slice_config['general']['data_dir'] / slice_config['inference']['data_dir'] / slice_config['inference']['masks_dir']
    labelled_mask_dir = labelled_mask_dir.as_posix()

    # get the model path from config
    model_path = ROOT / slice_config['model']['model_dir'] / slice_config['model']['model_name']
    model_path = model_path.as_posix()

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
    
    ###########################################################################################################

    ####################################### Functional Part of Program ########################################

    try:
        for filename in os.listdir(img_dir):
            try:
                img = cv2.imread(os.path.join(img_dir, filename), 1)   # read as BGR
                # img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)   # NOTE: test with this if RAM is too less to process large images
                input_img = preprocess_img(img, backbone)
            except Exception as e:
                logger.error("Image reading and preprocessing failed!")
                raise e

            try:
                labelled_mask = cv2.imread(os.path.join(labelled_mask_dir, filename), 0)   # read as grayscale
                # labelled_mask = cv2.resize(labelled_mask, (0, 0), fx = 0.5, fy = 0.5, interpolation = cv2.INTER_AREA)   # NOTE: test with this if RAM is too less to process large images
            except Exception as e:
                logger.error("Labelled masks for the test images do not exist at %s! If you wish to only infer on test images, run the 'inference.py'." % labelled_mask_dir)
                raise e

            try:
                model = load_model(model_path, compile=False)
            except Exception as e:
                logger.error("Could not load the model from %s!" % model_path)
                raise e

            try:
                # Model prediction using smooth blending
                # Use the algorithm. The `pred_func` is passed and will process all the image 8-fold by tiling small patches with overlap, called once with all those image as a batch outer dimension.
                # Note that model.predict(...) accepts a 4D tensor of shape (batch, x, y, nb_channels), such as a Keras model.
                predictions_smooth = predict_img_with_smooth_windowing(
                    input_img,
                    window_size=patch_size,
                    subdivisions=2,  # minimal amount of overlap for windowing: must be an even number
                    nb_classes=n_classes,
                    pred_func=(
                        lambda img_batch_subdiv: model.predict((img_batch_subdiv))
                    )
                )
            except Exception as e:
                logger.error("Model prediction failed!")
                raise e

            try:
                pred_mask = postprocess_mask(predictions_smooth)
            except Exception as e:
                logger.error("Post-processing predicted mask failed!")
                raise e

            try:
                # Save pred mask with file name and extension same as the input image and mask
                cv2.imwrite(os.path.join(pred_mask_dir, filename), pred_mask)
            except Exception as e:
                logger.error("Saving post-processed predicted mask failed!")
                raise e

            try:
                # Plot the predicted masks along with the original images and masks and save them
                plot_test(img, labelled_mask, pred_mask, pred_plot_dir, filename)
            except Exception as e:
                logger.error("Plotting the predicted mask along with the corresponding image and labelled mask and saving them failed!")
                raise e

    except Exception as e:
        logger.error("Images for testing do not exist at %s!" % img_dir)
        raise e

        ###########################################################################################################