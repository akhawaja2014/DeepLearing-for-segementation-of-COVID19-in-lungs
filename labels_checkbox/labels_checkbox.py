from PyQt5.QtWidgets import \
    QVBoxLayout, \
    QWidget, \
    QCheckBox, \
    QGroupBox

from PyQt5.QtCore import \
    Qt, \
    pyqtSignal

class LabelsCheckbox(QWidget):
    '''
    Check that handles a group of check boxes that decides which
    labels will be shown.

    When a checkbox state is changed, a signal is emitted with
    the active labels at that moment.
    '''
    state_changed = pyqtSignal()
    def __init__(self, labels, parent=None):
        '''
        Constructor

        Args:
            labels: List with the labels names. The position
                of each label should match with the label id in the mask.
        '''
        super(LabelsCheckbox, self).__init__(parent)
        self.check_box_widgets = []
        self.last_state = []
        self.initialize_widget(labels)

    def initialize_widget(self, labels):
        '''
        Initialize the widget configuration

        Args:
            labels: List with the labels names. The position
                of each label should match with the label id in the mask.
        '''
        group = QGroupBox('Labels shown')
        main_layout = QVBoxLayout()
        layout = QVBoxLayout()
        for label in labels:
            box = QCheckBox(label)
            box.setChecked(True)
            layout.addWidget(box)
            box.stateChanged.connect(self.on_state_changed)
            self.check_box_widgets.append(box)

        group.setLayout(layout)

        main_layout.addWidget(group)
        self.setLayout(main_layout)
        self.on_state_changed(0)

    def on_state_changed(self, state):
        '''
        Slot executed whenever one check box state is changed.
        It will check the state of all the checkboxes, and it will
        emit a signal with the IDs of the active ones.

        Args:
            state: For the changed checkbox, it is the actual value of the
                checkbox (Qt.Unchecked, Qt.PartiallyChecked, Qt.Checked)
        '''
        self.last_state = []
        for i, ch_box in enumerate(self.check_box_widgets):
            if ch_box.checkState() == Qt.Checked:
                self.last_state.append(i)

        self.state_changed.emit()