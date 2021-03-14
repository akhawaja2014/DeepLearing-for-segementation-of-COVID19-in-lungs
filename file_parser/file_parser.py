from PyQt5.QtCore import \
    QObject, \
    pyqtSignal

import nibabel as nib

import numpy as np
import os

class FileParser(QObject):
    '''
    This module encloses the Nifty format parser needed to read the MRI
    images. Since there exists already a module that does this, this class
    is really simple. It contains one slot that receives the filepath, and a
    signal that is emitted with the loaded image from that file.
    '''
    read_finished = pyqtSignal()

    def __init__(self, parent=None):
        super(FileParser, self).__init__(parent)
        self.images = None

    def request_open_file(self, filepath):
        if filepath and os.path.exists(filepath):
            image_data = self.load_nifty_file(filepath)
            self.images = self.correct_image(image_data)

            self.read_finished.emit()

    def load_nifty_file(self, filepath):
        '''
        Load a single nifty file.

        Args:
            filepath: Filepath of the file to be loaded.
                The path must exists.

        Returns:
            Data array with the set of images loaded from the file. If the file is
            3D, then another axis is added in order to have a 4D format, and do not
            treat the image differently from the 4D images.
        '''
        n1_img = nib.load(filepath)
        image_data = np.array(n1_img.get_fdata())
        return image_data

    def correct_image(self, img):
        '''
        It will change the intensity values in order to have values between 0 and 255

        TODO: Check if there is a parameter about the orientation of the image in the Nifty file

        Args:
            img: Set of images to be corrected

        Returns:
            Set of images, with the same shape as the input
        '''
        # We correct the negative values, so the image will be between 0 and the max value
        min_val = np.min(img)
        if min_val < 0:
            img = img + abs(min_val)

        # We ensure that the data is between 0 and 255
        max_val = np.max(img)
        if max_val:
            img = np.array(img * (255.0 / max_val), dtype=np.uint8)
        else:
            img = np.array(img, dtype=np.uint8)

        return img

    def close_file(self):
        '''
        Clear the internal buffers when closing the loaded images
        '''
        self.images = None