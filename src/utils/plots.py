import os
import matplotlib.pyplot as plt

def plot_test(img, labelled_mask, pred_mask, pred_plot_dir, filename):
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

def plot_inference(img, pred_mask, pred_plot_dir, filename):
    # Plot the predictions and save them
    plt.figure(figsize=(18, 4))
    # plotting the input image
    plt.subplot(121)
    plt.title('Test Image')
    plt.imshow(img)
    # plotting the predicted mask
    plt.subplot(122)
    plt.title('Predicted mask with smooth blending')
    plt.imshow(pred_mask)
    # saving and showing the plot
    plt.savefig(os.path.join(pred_plot_dir, filename.replace('.tif', '.png')))
    plt.show()