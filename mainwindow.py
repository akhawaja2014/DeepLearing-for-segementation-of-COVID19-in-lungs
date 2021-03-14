from PyQt5.QtWidgets import \
    QMainWindow, \
    QHBoxLayout, \
    QVBoxLayout, \
    QGridLayout, \
    QPushButton, \
    QWidget, \
    QLabel, \
    QScrollArea, \
    QShortcut, \
    QFileDialog, \
    QMessageBox, \
    QSpinBox

from PyQt5.QtGui import \
    QKeySequence

from PyQt5.QtCore import \
    Qt, \
    QCoreApplication

from image_widget.image_widget import ImageWidget
from file_parser.file_parser import FileParser
from image_segmentation.dl_segmentation_model import DLSegmenationModel
from labels_checkbox.labels_checkbox import LabelsCheckbox
from model_evaluation_widget.model_evaluation_widget import ModelEvaluationWidget

import copy
import os

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        '''
        Constructor. It will initialize the MainWindow layouts and its widgets
        '''
        super(MainWindow, self).__init__(parent)
        # The main layout will be splitted in 2 vertical layouts: Left and right.
        # Then each layout contains several widgets.
        self.main_layout = QHBoxLayout()
        self.left_layout = QVBoxLayout()
        self.right_layout = QVBoxLayout()

        self.source_image = None
        self.segm_image = None
        self.label_orig = None
        self.label_seg = None
        self.checkbox_widget = None
        self.is_image_loaded = False
        self.stored_segmentations = {}

        # Initialize DL model
        self.model = DLSegmenationModel()

        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_filepath = os.path.join(root_dir, 'image_segmentation',   'models', 'test_model.yaml')
        self.weights_filepath = os.path.join(root_dir, 'image_segmentation', 'models', 'test_model.h5')

        self.labels = self.model.used_classes

        self.initialize_widgets()
        self.setWindowTitle("Scene segmentation app")

    def initialize_widgets(self):
        # Parser that will be in charge of processing the opened files
        self.file_parser = FileParser()
        self.file_parser.read_finished.connect(self.on_image_loaded)

        # The initialization order matters. First we need to create the image widget
        # and then we add the buttons
        self.add_image_widget()
        self.add_buttons()
        self.add_spinbox()
        self.add_checkboxes()
        self.add_evaluation_widget()

        ## All the widget should be added to the main layout before this line
        self.set_main_window_layouts()

    def set_main_window_layouts(self):
        # We create the layouts that are going to host the left and right layouts
        self.left_widget = QWidget()
        self.right_widget = QWidget()
        self.left_widget.setLayout(self.left_layout)
        self.right_layout.addStretch()
        self.right_widget.setLayout(self.right_layout)

        # We add the left and right layouts to the main one
        self.main_layout.addWidget(self.left_widget)
        self.main_layout.addWidget(self.right_widget)
        self.internal_widget = QWidget()
        self.internal_widget.setLayout(self.main_layout)
        self.setCentralWidget(self.internal_widget)

    def add_buttons(self):
        '''
        Add the run and cancel buttons to the right side of the application,
        and the open and close file buttons to the left
        '''
        self.run_button = QPushButton('RUN')
        self.cancel_button = QPushButton('Cancel')
        self.open_button = QPushButton('Open file...')
        self.close_button = QPushButton('Close file...')

        self.right_layout.addWidget(self.run_button)
        self.right_layout.addWidget(self.cancel_button)
        self.left_layout.addWidget(self.open_button)
        self.left_layout.addWidget(self.close_button)

        # Shortcuts
        self.run_button.setToolTip("Run (Enter)")
        shortcut = QShortcut(QKeySequence(Qt.Key_Return), self)
        shortcut.activated.connect(self.on_run_segmentation)

        self.cancel_button.setToolTip("Cancel (Esc)")
        shortcut = QShortcut(QKeySequence(Qt.Key_Escape), self)
        shortcut.activated.connect(self.on_cancel)

        self.open_button.setToolTip("Open a new file (Ctrl + O)")
        shortcut = QShortcut(QKeySequence("Ctrl+o"), self)
        shortcut.activated.connect(self.on_click_load_image)

        # Connections
        self.run_button.clicked.connect(self.on_run_segmentation)
        self.cancel_button.clicked.connect(self.on_cancel)
        self.open_button.clicked.connect(self.on_click_load_image)
        self.close_button.clicked.connect(self.on_click_close_image)

    def add_image_widget(self):
        '''
        Add the widget where we are going to show the image.
        It will be added in the left side of the main window.
        The image widget is located in an Scrollable area, so it does not
        matter the size of the screen, we can always see all the images
        from the opened MRI file
        '''
        layout = QGridLayout()
        image_widget = QWidget()
        self.message_label = QLabel('No image loaded...')
        self.left_layout.addWidget(self.message_label)

        self.source_image = ImageWidget()
        self.segm_image = ImageWidget()

        layout.addWidget(self.source_image, 0, 0)
        layout.addWidget(self.segm_image, 1, 0)
        self.label_orig = QLabel('Original')
        self.label_seg = QLabel('Segmented')
        layout.addWidget(self.label_orig, 0, 1)
        layout.addWidget(self.label_seg, 1, 1)

        image_widget.setLayout(layout)

        self.scroll = QScrollArea()
        self.scroll.setWidget(image_widget)
        self.scroll.setWidgetResizable(True)
        self.left_layout.addWidget(self.scroll)
        self.label_orig.hide()
        self.label_seg.hide()

    def add_spinbox(self):
        '''
        Add a spinbox that will help to switch from one slide to another one,
        just changing a number. This widget will be added in the right
        layout
        '''
        self.spinbox = QSpinBox()
        self.spinbox.valueChanged.connect(self.on_new_slide_selected)
        self.spinbox.setMinimum(0)
        self.spinbox.setMaximum(0)
        self.spinbox_label = QLabel("Slice number")
        layout = QHBoxLayout()
        custom_widget = QWidget()
        layout.addWidget(self.spinbox_label)
        layout.addWidget(self.spinbox)
        custom_widget.setLayout(layout)
        self.right_layout.addWidget(custom_widget)

    def add_checkboxes(self):
        self.checkbox_widget = LabelsCheckbox(self.labels)
        self.right_layout.addWidget(self.checkbox_widget)
        self.checkbox_widget.state_changed.connect(self.on_chbox_state_changed)

    def add_evaluation_widget(self):
        self.evaluation_widget = ModelEvaluationWidget(self.model)
        self.right_layout.addWidget(self.evaluation_widget)

    def set_buttons_enabled(self, enabled):
        '''
        Enable or disable all the buttons at once, except for the cancel
        button.
        We don't change the cancel button since it will be the only way to
        abort the processing when the segmentation is running

        Args:
            enabled: Boolean. If True, this function will enable all the buttons,
                and it will disable them if this argument if False.
        '''
        self.run_button.setEnabled(enabled)
        self.open_button.setEnabled(enabled)
        self.close_button.setEnabled(enabled)
        self.spinbox.setEnabled(enabled)
        self.checkbox_widget.setEnabled(enabled)

    # Slots
    def on_cancel(self):
        '''
        Slot to be executed when the cancel button is pressed.
        If an image is loaded, this function will enable the other buttons
        '''
        self.evaluation_widget.cancel_test()
        if self.is_image_loaded:
            self.set_buttons_enabled(True)

    def on_run_segmentation(self):
        '''
        Slot executed when the RUN button is pressed.
        If an image has been loaded, this function will execute the segmentation directly.

        When finished, it will re enable the buttons of the MainWindow and
        restore the image view
        '''
        if self.is_image_loaded:
            # We load the right model
            if not self.model.load_model(self.model_filepath, self.weights_filepath):
                print("CRITICAL! Cannot load the DeepLearning model at {}".format(self.model_filepath))
                assert(0)

            msg = QMessageBox()
            msg.setIcon(QMessageBox.Information)
            msg.setText("Segmentation in progress. This operation may take a moment")
            msg.setWindowTitle("Running segmentation")
            msg.setWindowModality(Qt.NonModal)
            msg.setStandardButtons(QMessageBox.NoButton)
            msg.show()
            self.set_buttons_enabled(False)

            # We update the GUI events before and after the segmentation,
            # to reduce the GUI lagging
            QCoreApplication.processEvents()

            idx = self.spinbox.value()
            detected_image = self.model.estimate_image(self.file_parser.images[:,:,idx])

            QCoreApplication.processEvents()

            # TODO: Make the array [0,1,2] to be a set of check boxes where the user
            # can set which labels he wants to see
            self.segm_image.set_new_masks(detected_image, self.checkbox_widget.last_state)
            self.stored_segmentations[idx] = detected_image
            self.set_buttons_enabled(True)
            msg.close()

    def on_click_load_image(self):
        '''
        Slot executed when the Open file button is pressed.
        It will show a file dialog where the user has to choose a file.
        If a valid string is provided, then the filepath is provided to the
        file parser to try to process the file.
        '''
        fname = QFileDialog.getOpenFileName(self,
            'Open file',
            '',
            "All files ( *.* )")
        if fname[0]:
            self.on_click_close_image()
            self.file_parser.request_open_file(fname[0])
        else:
            print("No path value entered.")

    def on_click_close_image(self):
        '''
        Slot executed when we press the Close file button.
        It will clear the local image buffers and it will show the label of
        "No image loaded"
        '''
        self.is_image_loaded = False
        self.stored_segmentations.clear()
        self.spinbox.setMaximum(0)
        self.source_image.clear_local_buffers()
        self.segm_image.clear_local_buffers()
        self.message_label.show()
        self.label_orig.hide()
        self.label_seg.hide()
        self.file_parser.close_file()

    def on_image_loaded(self):
        '''
        Slot executed when the file parser opened a file successfully.
        When this slot is executed, we can query the image information,
        in order to show it in the GUI
        '''
        self.source_image.clear_local_buffers()
        self.segm_image.clear_local_buffers()

        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText("Loading files. This operation may take a moment")
        msg.setWindowTitle("Loading files")
        msg.setWindowModality(Qt.NonModal)
        msg.setStandardButtons(QMessageBox.NoButton)
        msg.show()

        # TODO: Optional. Add preprocessing the image here

        self.is_image_loaded = True
        self.label_orig.show()
        self.label_seg.show()
        self.message_label.hide()
        self.spinbox.setMaximum(self.file_parser.images.shape[2] - 1)
        self.on_new_slide_selected(0)
        msg.close()

    def on_new_slide_selected(self, val):
        '''
        Slot executed whenever the user changes the value of the spinbox.
        It will update the shown image

        Args:
            val: New value set for the spinbox
        '''
        if self.is_image_loaded:
            self.source_image.update_image(self.file_parser.images[:,:,val])
            self.segm_image.update_image(self.file_parser.images[:,:,val])
            old_mask = self.stored_segmentations.get(val)
            if not old_mask is None:
                self.segm_image.set_new_masks(old_mask, self.checkbox_widget.last_state)

    def on_chbox_state_changed(self):
        '''
        Slot executed whenever one checkbox state is changed.
        '''
        if self.is_image_loaded:
            idx = self.spinbox.value()
            old_mask = self.stored_segmentations.get(idx)
            if not old_mask is None:
                self.segm_image.set_new_masks(old_mask, self.checkbox_widget.last_state)