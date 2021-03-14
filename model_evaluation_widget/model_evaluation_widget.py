from PyQt5.QtWidgets import \
    QVBoxLayout, \
    QPushButton, \
    QWidget, \
    QFileDialog, \
    QMessageBox

from PyQt5.QtCore import \
    Qt, \
    QCoreApplication

import glob
import os
import nibabel as nib
import numpy as np
import cv2
import copy

from scipy.spatial.distance import directed_hausdorff
from scipy.stats import pearsonr
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

class ModelEvaluationWidget(QWidget):
    def __init__(self, model_to_assess, parent=None):
        super(ModelEvaluationWidget, self).__init__(parent)
        self.model_to_assess = model_to_assess
        assert(not self.model_to_assess is None)

        self.patient_folder_pattern = 'cases'
        self.ground_truth_folder_pattern = 'masks'
        self.nifty_file_pattern = '.nii.gz'
        # We specify the size to which we will convert all the images
        self.resized_image_size = (256,256)
        # We specify where the models are
        dirname = os.path.dirname(os.path.abspath(__file__))
        self.models_path = os.path.join(dirname, '..', 'image_segmentation', 'models', 'last_efficient_18images')

        self.is_running = False
        self.dice_coeffs = []
        self.hausdorff_distances = []

        self.initialize_widget()

    def initialize_widget(self):
        layout = QVBoxLayout()
        button = QPushButton("Evaluate model")
        button.clicked.connect(self.evaluate_model)
        layout.addWidget(button)
        self.setLayout(layout)

    def search_nifty_files(self, source_path):
        # Returns a list of names in list files.
        files = glob.glob(os.path.join(source_path, '**'), recursive = True)
        dirnames = []
        # Get directories that contains nifty files, without dupplications
        for file in files:
            if os.path.isfile(file) and self.nifty_file_pattern in file:
                directory = os.path.dirname(file)
                if not directory in dirnames:
                    dirnames.append(directory)

        # Sanity check: We only should have two folders in this place: Cases and masks
        if len(dirnames) == 0:
            print("The folder {d} is empty or it does not exists".format(d=source_path))
            return
        else:
            assert(len(dirnames) == 2)

        if 'cases' in dirnames[0]:
            patients_folder = dirnames[0]
            gt_folder = dirnames[1]
        else:
            patients_folder = dirnames[1]
            gt_folder = dirnames[0]

        # We search for files in those directories, and we split them in two groups,
        # patient and gt
        self.patient_files = []
        self.gt_files = []
        cases_files = glob.glob(os.path.join(patients_folder, '**'), recursive = False)
        masks_files = glob.glob(os.path.join(gt_folder, '**'), recursive = False)

        for case in cases_files:
            case_filename = os.path.basename(case)
            expected_mask_file = os.path.join(gt_folder, case_filename)
            if expected_mask_file in masks_files:
                self.patient_files.append(case)
                self.gt_files.append(expected_mask_file)

        print("{n} Nifty files found".format(n=len(self.patient_files)))

    def read_image(self, filepath):
        '''
        Read a Nifty file, and re scale the intensities so it
        has values between 0 and 255

        Args:
            filepath: Path where the file to be opened lies

        Returns:
            Array with the image information
        '''
        if filepath and os.path.exists(filepath):
            n1_img = nib.load(filepath)
            image_data = np.array(n1_img.get_fdata())
            # The frames must be only at one time (It is not 4D image)
            assert(len(image_data.shape) == 3)
            min_val = np.min(image_data)
            if min_val < 0:
                image_data = image_data + abs(min_val)

            # We ensure that the data is between 0 and 255
            max_val = np.max(image_data)
            if max_val:
                image_data = np.array(image_data * (255.0 / max_val), dtype=np.uint8)
            else:
                image_data = np.array(image_data, dtype=np.uint8)

            return image_data
        else:
            return np.array([])

    def cancel_test(self):
        self.is_running = False

    def compute_single_dice(self, pred_mask, gt_mask):
        intersection = np.array(np.logical_and(gt_mask, pred_mask), dtype=np.uint8)
        pred_count = np.array(np.where(pred_mask == 1)).shape[1]
        gt_count = np.array(np.where(gt_mask == 1)).shape[1]
        intersection_count = np.array(np.where(intersection == 1)).shape[1]

        if gt_count == 0 and pred_count == 0:
            return 1.0
        else:
            return 2.0 * intersection_count / (gt_count + pred_count)

    def compute_dice_coeff(self, predicted, gt):
        gt_right_lung = np.array(gt == 85, dtype=np.uint8)
        gt_left_lung = np.array(gt == 170, dtype=np.uint8)
        gt_disease = np.array(gt == 255, dtype=np.uint8)

        pred_right_lung = np.array(predicted == 85, dtype=np.uint8)
        pred_left_lung = np.array(predicted == 170, dtype=np.uint8)
        pred_disease = np.array(predicted == 255, dtype=np.uint8)

        rl_dice = self.compute_single_dice(pred_right_lung, gt_right_lung)
        ll_dice = self.compute_single_dice(pred_left_lung, gt_left_lung)
        dis_dice = self.compute_single_dice(pred_disease, gt_disease)

        return (rl_dice, ll_dice, dis_dice)

    def get_hausforff_dist(self, pred_mask, gt_mask):
        # Taken from
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.directed_hausdorff.html
        return max(directed_hausdorff(pred_mask, gt_mask)[0], directed_hausdorff(gt_mask, pred_mask)[0])

    def compute_hausforff_distances(self, predicted, gt):
        gt_right_lung = np.array(gt == 85, dtype=np.uint8)
        gt_left_lung = np.array(gt == 170, dtype=np.uint8)
        gt_disease = np.array(gt == 255, dtype=np.uint8)

        pred_right_lung = np.array(predicted == 85, dtype=np.uint8)
        pred_left_lung = np.array(predicted == 170, dtype=np.uint8)
        pred_disease = np.array(predicted == 255, dtype=np.uint8)

        rl_hausforff = self.get_hausforff_dist(pred_right_lung, gt_right_lung)
        ll_hausforff = self.get_hausforff_dist(pred_left_lung, gt_left_lung)
        dis_hausforff = self.get_hausforff_dist(pred_disease, gt_disease)

        return (rl_hausforff, ll_hausforff, dis_hausforff)

    def plot_roc(self, pred_gt_list, fig_name):
        # We create a list with the label values:
        # 85 --> right lung
        # 170 --> left lung
        # 255 --> Disease
        right_lung_dict = {
            'label_val': 85,
            'plot_title': 'Right lung overall',
            'plot_color': '-b',
        }
        left_lung_dict = {
            'label_val': 170,
            'plot_title': 'Left lung overall',
            'plot_color': '-g',
        }
        disease_dict = {
            'label_val': 255,
            'plot_title': 'Disease overall',
            'plot_color': '-r',
        }

        dictionaries = [right_lung_dict, left_lung_dict, disease_dict]
        np_pred_gt_list = np.array(pred_gt_list)

        plt.figure()
        line_width = 1
        for class_dict in dictionaries:
            correct_masks = np.array(np_pred_gt_list == class_dict['label_val'], dtype=np.uint8)
            pred_images = correct_masks[:,0]
            gt_images = correct_masks[:,1]
            flattenGT = np.stack(gt_images, axis=0).ravel()
            flattenPD = np.stack(pred_images, axis=0).ravel()
            fpr_f1, tpr_f1, thresholds_fcn = roc_curve(flattenGT, flattenPD)
            auc_f = roc_auc_score(flattenGT, flattenPD)
            # plot perfromance for binary segemnatation
            plt.grid(True)
            plt.plot(
                fpr_f1,
                tpr_f1,
                class_dict['plot_color'],
                label= class_dict['plot_title'] + ' - AUC = ' + str(round(auc_f, 3)))

        plt.legend(loc = 'lower right')
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.plot([0, 1], [0, 1], color='navy', lw=line_width, linestyle='--')

        dirname = os.path.dirname(os.path.abspath(__file__))
        plt.savefig(os.path.join(dirname, 'results', fig_name + '_roc_auc_curves.png'))
        plt.close()

    def autolabel(self, rects):
        # Taken from https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
        """Attach a text label above each bar in *rects*, displaying its height."""
        ax = plt.gca()
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    def get_histogram(self, data, amount_of_bins, max_value):
        # Sanity check
        assert(amount_of_bins > 0)
        np_data = np.array(data)
        print(np_data)

        increase_factor = max_value / int(amount_of_bins + 0.5)
        bins = []
        x_axis = []

        count = np.array(np.where((np_data >= 0) & (np_data <= increase_factor))).shape[1]
        x_axis.append(0)
        bins.append(count)

        for i in range(1, amount_of_bins):
            min_bin_val = i * increase_factor
            max_bin_val = (i+1) * increase_factor
            count = np.array(np.where((np_data > min_bin_val) & (np_data <= max_bin_val))).shape[1]
            x_axis.append(min_bin_val)
            bins.append(count)

        x_axis  = np.array(x_axis) + (increase_factor / 2.0)
        return x_axis, bins

    def plot_dice_histogram(self, dice_coeff, amount_of_bins, fig_name):
        # We create a list with the label values:
        right_lung_dict = {
            'label': 'Right lung',
            'output_file': 'dice_right_lung.png',
        }
        left_lung_dict = {
            'label': 'Left lung',
            'output_file': 'dice_left_lung.png',
        }
        disease_dict = {
            'label': 'Disease',
            'output_file': 'dice_disease.png',
        }
        dictionaries = [right_lung_dict, left_lung_dict, disease_dict]
        np_dice_coeff = 100.0 * np.array(dice_coeff)
        dirname = os.path.dirname(os.path.abspath(__file__))

        for i, class_dict in enumerate(dictionaries):
            plt.figure()
            # We know that the dice coeff is between 0 and 100, so we set the max value
            # to 100 directly
            x_axis, hist = self.get_histogram(np_dice_coeff[:,i], amount_of_bins, max_value=100)
            # We set a width smaller than the bar width, so we can see when a
            # bar starts, and when the other finishes
            bar_width = 0.8 * (x_axis[1] - x_axis[0])
            plt.grid(True)
            rects = plt.bar(x_axis, hist, width=bar_width, label=class_dict['label'])
            self.autolabel(rects)
            plt.gca().legend()
            plt.xticks(range(0, 101,5))
            # We extend the Y limit, so we can clearly see the bar value of the biggest bar
            plt.ylim([0, np.max(hist) * 1.2])
            plt.savefig(os.path.join(dirname, 'results', fig_name + '_' + class_dict['output_file']))
            plt.close()

    def plot_hausdorff_histogram(self, hausdorff_dist, amount_of_bins, fig_name):
        # We create a list with the label values:
        right_lung_dict = {
            'label': 'Right lung',
            'output_file': 'hausdorff_right_lung.png',
        }
        left_lung_dict = {
            'label': 'Left lung',
            'output_file': 'hausdorff_left_lung.png',
        }
        disease_dict = {
            'label': 'Disease',
            'output_file': 'hausdorff_disease.png',
        }
        dictionaries = [right_lung_dict, left_lung_dict, disease_dict]
        np_hausdorff_dist = np.array(hausdorff_dist)
        dirname = os.path.dirname(os.path.abspath(__file__))

        for i, class_dict in enumerate(dictionaries):
            plt.figure()
            correct_h_distance = np_hausdorff_dist[:,i]
            # Since there is no limit for the Hausdorff distance, we set the maximum value of the
            # X axis to the max distance found
            x_axis, hist = self.get_histogram(correct_h_distance, amount_of_bins, max_value=np.max(correct_h_distance))
            # We set a width smaller than the bar width, so we can see when a
            # bar starts, and when the other finishes
            bar_width = 0.8 * (x_axis[1] - x_axis[0])
            plt.grid(True)
            rects = plt.bar(x_axis, hist, width=bar_width, label=class_dict['label'])
            self.autolabel(rects)
            plt.gca().legend()
            # We extend the Y limit, so we can clearly see the bar value of the biggest bar
            plt.ylim([0, np.max(hist) * 1.2])
            plt.savefig(os.path.join(dirname, 'results', fig_name + '_' + class_dict['output_file']))
            plt.close()

    def stats_analysis(self, pred_gt_list, name):
        self.dice_coeffs = []
        self.hausdorff_distances = []
        for (pred, gt) in pred_gt_list:
            self.dice_coeffs.append(self.compute_dice_coeff(pred, gt))
            self.hausdorff_distances.append(self.compute_hausforff_distances(pred, gt))
            QCoreApplication.processEvents()

        self.plot_roc(pred_gt_list, fig_name=name)
        self.plot_dice_histogram(self.dice_coeffs, amount_of_bins=20, fig_name=name)
        self.plot_hausdorff_histogram(self.hausdorff_distances, amount_of_bins=20, fig_name=name)

    def search_models(self, source_path):
        # Returns a list of names in list files.
        dirnames = []
        h5_pattern = '.h5'
        yaml_pattern = '.yaml'

        self.retrieved_h5_files = []
        self.retrieved_yaml_files = []
        h5_files = glob.glob(os.path.join(source_path, '*.h5'), recursive = False)
        yaml_files = glob.glob(os.path.join(source_path, '*.yaml'), recursive = False)

        while len(yaml_files) and len(h5_files):
            filename = os.path.splitext(os.path.basename(yaml_files[0]))[0]
            dirname = os.path.dirname(yaml_files[0])

            # We build the two filespath that we expect to found
            h5_file = os.path.join(dirname, filename + h5_pattern)
            yaml_file = os.path.join(dirname, filename + yaml_pattern)

            # We only save the file model name if h5 and yaml files are found
            if h5_file in h5_files and yaml_file in yaml_files:
                self.retrieved_yaml_files.append(yaml_file)
                self.retrieved_h5_files.append(h5_file)
                h5_files.remove(h5_file)

            yaml_files.remove(yaml_files[0])

        print("{n} DeepLearning models found".format(n=len(self.retrieved_yaml_files)))

    def evaluate_model(self):
        self.is_running = True

        # We search all the pairs of YAML and h5 files that are in this folder
        self.search_models(self.models_path)

        # We query the user to decide where the testing files are (images and GT masks)
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText("Select the folder where the masks and the source files are")
        msg_box.setWindowTitle("Evaluating the model")
        msg_box.setWindowModality(Qt.NonModal)
        msg_box.setModal(True)
        msg_box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        pressed = msg_box.exec()
        if pressed == QMessageBox.Cancel:
            self.is_running = False
            return

        folder_name = ''
        dialog = QFileDialog()
        dialog.setFileMode(QFileDialog.DirectoryOnly)
        dialog.setOption(QFileDialog.ShowDirsOnly, False)
        if(dialog.exec()):
            folder_name = dialog.directory().absolutePath()
        else:
            print("No path value entered.")
            self.is_running = False
            return

        # We load in RAM all the images (patient images and labeled GT images)
        self.search_nifty_files(folder_name)
        self.pat_images = []
        self.gt_images = []
        for pat_fp, gt_fp in zip(self.patient_files, self.gt_files):
            if not self.is_running:
                return

            print("Storing file {}".format(os.path.basename(pat_fp)))
            pat = self.read_image(pat_fp)
            gt = self.read_image(gt_fp)
            for i in range(gt.shape[2]):
                if not self.is_running:
                    return

                source_img = pat[:,:,i]
                gt_img = gt[:,:,i]
                self.pat_images.append(cv2.resize(source_img, self.resized_image_size, interpolation = cv2.INTER_AREA))
                self.gt_images.append(cv2.resize(gt_img, self.resized_image_size, interpolation = cv2.INTER_AREA))

                QCoreApplication.processEvents()

        self.gt_images = np.array(self.gt_images)
        self.pat_images = np.array(self.pat_images)
        # Sanity check for data consistency
        assert(self.gt_images.shape[0] == self.pat_images.shape[0])

        print("We will analysize {} images".format(self.gt_images.shape[0]))
        for i in range(len(self.retrieved_h5_files)):
            if not self.is_running:
                return

            # We load the corresponding model
            if not self.model_to_assess.load_model(self.retrieved_yaml_files[i], self.retrieved_h5_files[i]):
                print("CRITICAL! Cannot load the DeepLearning model at {}".format(model_filepath))
                assert(0)

            # We extract the filename, so the stored files will contain this name too
            model_name = os.path.splitext(os.path.basename(self.retrieved_yaml_files[i]))[0]
            self.predicted_gt_list = []

            # We loop over all the retrieved images, to estimate the masks, and then compute the stats
            image_range = range(self.gt_images.shape[0])
            for i in image_range:
                if not self.is_running:
                    return
                QCoreApplication.processEvents()

                print("Estimating mask #{}".format(i + 1))
                img = copy.deepcopy(self.pat_images[i])
                gt  = copy.deepcopy(self.gt_images[i])
                # We store a pair predicted_mask - GT mask
                # We know that the masks that predict returns can have
                # values between 0 and 3, so the scaling factor to have
                # values between 0 and 255 is 255 / 3
                predicted = np.array(self.model_to_assess.estimate_image(img) * (255.0 / 3.0), dtype=np.uint8)
                self.predicted_gt_list.append((predicted, gt))

            if self.is_running:
                self.stats_analysis(self.predicted_gt_list, name=model_name)

        print("Test finished!!")
        self.is_running = False
