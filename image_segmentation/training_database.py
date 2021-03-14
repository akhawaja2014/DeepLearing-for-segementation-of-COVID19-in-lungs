import os
import nibabel as nib
import numpy as np
import cv2

class TrainingDataset(object):
    '''
    This class is aimed to store the images from the cases and the corresponding GT
    images.

    The dataset can be always expanded. A gt file and its corresponding image filepaths
    must be provided. The loaded images can be queried using the get_training_data function
    which will give 4 lists of images: Training, Labels of training, validation, labels of validation

    The supported files must be Nifty, and they should be splitted into two folders:
    cases and masks. The names can contain other strings, but they must contain
    those words in their names in order to be detected.

    If other folders with Nifty files are located in the same folder as the masks and
    the cases, an assertion will be thrown.
    '''
    def __init__(self, storage_shape):
        self.case_images = []
        self.mask_images = []
        self.image_size = storage_shape

    def reset_buffers(self):
        '''
        Clear the internal bufffers of loaded images
        '''
        self.case_images.clear()
        self.mask_images.clear()

    def append_data(self, image, gt_file):
        '''
        Add more data to the dataset. Two files are required, patient data
        and the corresponding labeled image. They will be read, rescaled to
        the size provided in construction, and stored into internal arrays.

        The provided files must be given in Nifty format, of extension nii.gz
        Args:
            image: filepath of the patient nifty file
            gt_file: filepath of the corresponding labeled image
        '''
        print("Loading file: {f}".format(f=os.path.basename(image)))

        gt_image = self.read_image(gt_file)
        assert(len(gt_image))
        patient_image = self.read_image(image)
        assert(len(patient_image))

        # Sanity check: Both images should have the same amount of slices
        assert(gt_image.shape[2] == patient_image.shape[2])
        for i in range(gt_image.shape[2]):
            scaled_patient_image = cv2.resize(patient_image[:,:,i], self.image_size, interpolation = cv2.INTER_AREA)
            scaled_gt_image = cv2.resize(gt_image[:,:,i], self.image_size, interpolation = cv2.INTER_AREA)
            self.case_images.append(scaled_patient_image)
            self.mask_images.append(scaled_gt_image)

    def get_training_data(self, training_ratio):
        '''
        Get a training set composed by 4 lists of images:
            -) Training patient data
            -) Training labeled images
            -) Validation patient data
            -) Validation labeled images

        The amount of data in each group is regulated by the provided ratio
        (value between 0 and 1), that tells which percentage of input images
        will correspond to the training dataset

        Args:
            training_ratio: Float between 0 and 1 that tells which percentage
                of the input data will be put as training data

        Returns:
            Four lists (X, Y, val_X, val_Y) where
                X = Training patient data
                Y = Training labeled images
                val_X = Validation patient data
                val_Y = Validation labeled images
        '''
        # Sanity check: The input training ratio must be a value between 0 and 1
        assert(training_ratio >=0 and training_ratio <= 1)
        print("Shape of requested data: {}".format(np.array(self.case_images).shape))

        training_data = np.array(self.case_images)
        label_data = np.array(self.mask_images)
        amount_of_data = training_data.shape[0]

        indices = np.arange(amount_of_data)
        indices = np.random.permutation(indices)

        training_data = training_data[indices]
        label_data = label_data[indices]

        split_index = int(amount_of_data * training_ratio + 0.5)
        return np.expand_dims(training_data[:split_index],axis=3), label_data[:split_index], np.expand_dims(training_data[split_index:],axis=3), label_data[split_index:]

    def get_cases_images(self):
        return self.case_images

    def get_masks_images(self):
        return self.mask_images

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