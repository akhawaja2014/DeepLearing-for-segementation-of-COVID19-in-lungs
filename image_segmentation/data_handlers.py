# Implementation taken mainly from
# https://github.com/qubvel/segmentation_models/blob/master/examples/multiclass%20segmentation%20(camvid).ipynb
import keras
from keras.utils import Sequence
import numpy as np

class Dataset(object):
    '''
    Args:
        input: List of source images matrices
        labels: List of corresponding label images of input
        classes: List of values of classes to extract from segmentation mask.
            The possible values are:
                'background'
                'leftlung'
                'rightlung'
                'disease'

        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    '''

    CLASSES = {
        'background': 0,
        'leftlung': 85,
        'rightlung': 170,
        'disease': 255,
    }

    def __init__(
            self,
            input,
            labels,
            classes=None,
            augmentation=None,
            preprocessing=None,
    ):
        self.input_dataset = input
        self.labels_dataset = labels

        # convert str names to class values on masks
        self.class_values = [self.CLASSES[cls.lower()] for cls in classes]

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        '''
        Overloads the operator []. It provides the image and corresponding label image
        of the element i of the input datasets, passed on contruction
        '''
        # read data
        image = self.input_dataset[i]
        mask = self.labels_dataset[i]

        # extract certain classes from mask (e.g. disease)
        masks = [(mask == v) for v in self.class_values]
        mask = np.stack(masks, axis=-1).astype('float')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        return image, mask

    def __len__(self):
        '''
        This value will be returned with the len() method is called
        '''
        return len(self.input_dataset)

class Dataloader(keras.utils.Sequence):
    '''
    Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch. By default it is set to 1
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    '''

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        '''
        Overloads the operator []. It provides a batch of images
        and their corresponding labels. It contains two levels of
        indexation: The first one is the index of the batch, the second
        one can be zero, for the source image, or 1 for the labels

        Args:
            i: Index where to start the batch

        Returns:
            List of lists. The first index will have the length of the batch_size
            passed on construction.
        '''
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        # TODO: Improve this. From a point of view of performance, this is horrible.
        # The best thing to do is to inherit dataset from Numpy array, and do indexing
        # instead of a for loop
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        return batch

    def __len__(self):
        '''
        Overloads the len operator. It denotes the number of batches per epoch
        '''
        # NOTE: // means Floor division in Python
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)