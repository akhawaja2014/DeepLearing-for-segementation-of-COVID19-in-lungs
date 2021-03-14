'''
The idea of this module is to focus on showing the image only of the loaded nifty file.
It will be a subclass of QWidget that reimplements the paintEvent method, so we can set
the image we want
'''

from PyQt5.QtWidgets import \
    QWidget

from PyQt5.QtGui import \
    QImage, \
    qRgb, \
    QPixmap, \
    QPainter, \
    QPolygonF, \
    QPen

from PyQt5.QtCore import \
    QPointF, \
    Qt, \
    QRectF

import copy
import numpy as np
import cv2

class ImageWidget(QWidget):
    '''
    This class is aimed to show a MR images at one depth.
    It also counts with the capability of showing the contours of the segmentation,
    when it is provided through the right function.
    '''
    def __init__(self, parent=None):
        super(ImageWidget, self).__init__(parent)
        self.image_set = None
        self.current_image = None
        self.current_data = None
        self.segm_masks_list = []
        self.size_image_set = 0
        self.scaling_factor = 1.0

        self.seg_mask_colors = [
            [255,255,0], # yellow
            [0,0,255],   # blue
            [255,0,0],   # red
            [0,255,255], # cyan
            [0,128,0],   # green
        ]

    def update_image(self, image_array):
        '''
        It updates the internal image buffer. Then, we show the widget.

        Args:
            image_array: New image to be set
        '''
        if not image_array is None and image_array.shape:
            # We copy the data into the internal buffer
            self.reset_masks()
            self.image_set = copy.deepcopy(image_array)
            self.show()
            self.refresh_image()

    def refresh_image(self):
        '''
        Refresh the image that is being shown. If masks are stored, then the image
        is updated with the label information overlapped.
        Then, only one image is generated here.
        '''
        if not self.image_set is None and self.image_set.shape:
            assert (np.max(self.image_set) <= 255)
            image8 = np.array(self.image_set[:,:]).astype(np.uint8, order='C', casting='unsafe')

            # We clear all the spots where we have labels
            for i in range(len(self.segm_masks_list)):
                clear_mask = self.segm_masks_list[i][0]
                # clear the bits where the mask goes
                image8 = np.multiply(image8, clear_mask)

            # We convert the image into RGB
            image8 = np.array(cv2.cvtColor(image8, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
            for i in range(len(self.segm_masks_list)):
                color_mask = np.array(self.segm_masks_list[i][1], dtype=np.uint8)
                # We color the area where the mask is
                image8 = image8 + color_mask

            height = image8.shape[0]
            width = image8.shape[1]

            # QImage hold a shallow copy of the data, so we need to keep the information while showing it.
            # We do this making a copy of the data and keeping it in the class
            self.current_data = copy.deepcopy(image8)
            # We create a QImage as an indexed image to show the grayscale
            # values. Because it is in indexed format, we set its color table
            # too to be grayscale
            qimage = QImage(
                self.current_data.data,
                width,
                height,
                3 * width,
                QImage.Format_RGB888)

            # We scale the image
            self.current_image = qimage.scaledToWidth(
                self.image_set.shape[1] * self.scaling_factor)

            # We limit the QWidget size to be the equal to the Image size
            self.setFixedSize(
                self.current_image.width(),
                self.current_image.height())

            self.repaint()

    def clear_local_buffers(self):
        '''
        Erase all the internal information stored.
        '''
        self.reset_masks()
        self.image_set = None
        self.current_image = QImage(
            np.array([]),
            0,
            0,
            0,
            QImage.Format_Indexed8)
        self.repaint()

    def set_new_masks(self, seg_mask, labels_to_show):
        '''
        Set the masks to be shown with the image. It will create a set of masks,
        with the information of each mask separated. The size of this set
        is given by the labels_to_show list. After the label has been updated,
        a refresh of the shown image is called.

        Args:
            seg_mask: Image with all the labels placed at the same time.
            labels_to_show: List with the labels values that we can find in image.
                The values find in this list are the labels that are going to be shown.
                For instance, we can have 4 labels, but if labels_to_show contains
                only one label, then only this label will be shown.
        '''
        self.reset_masks()
        for label in labels_to_show:
            mask = np.array(seg_mask == label, dtype=np.uint8)
            # We create a mask with the color of the type of object we want to segment
            color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) * self.seg_mask_colors[label]
            # In order to set the right color in the place, we need a mask that
            # clears what is in the place where we want to place the mask
            clearing_mask = mask = np.array(seg_mask != label, dtype=np.uint8)
            self.segm_masks_list.append([clearing_mask, color_mask])

        self.refresh_image()

    def reset_masks(self):
        self.segm_masks_list.clear()

    def get_image(self):
        return self.image_set

    def paintEvent(self, event):
        '''
        Overloaded paintEvent. It will draw the MRI image.
        '''
        painter = QPainter(self)
        if not self.current_image is None:
            painter.drawPixmap(0,0, self.current_image.width(), self.current_image.height(), QPixmap(self.current_image))