import sys
import os
dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(dirname, '..'))
from image_segmentation.data_handlers import Dataset, Dataloader

import os
# Libraries for CNN
from keras.models import Model, model_from_yaml
from keras.layers import concatenate, \
    Conv2D, \
    Conv2DTranspose, \
    Input, \
    MaxPooling2D, \
    BatchNormalization, \
    Activation, \
    Dropout
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from segmentation_models.losses import DiceLoss
import numpy as np
import cv2

class DLSegmenationModel(object):
    '''
    Class that creates a DL model for multi-class segmentation
    '''
    def __init__(self, epochs = 20, batch_size = 8, shuffle = True):
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.model = None

        # TODO: Change these functions when tuning the model
        self.loss_func = DiceLoss()
        self.optimizer = Adam(lr = 0.0001)
        self.used_classes = ['background', 'leftlung', 'rightlung', 'disease']
        self.resized_image_size = (256,256)

    def train(self, input_images):
        '''
        Train the DL model. This function will take the input images, and it will
        split them into training and validation, preprocess them, create the model,
        and train it.

        Args:
            input_images: TrainingDatabase object with all the loaded images and their
                corresponding labels


        TODO LIST: Points to check:
            *) batch_size
                -) DONE: This value regulates how many pictures are taken at each iteration.
                    The more batches, the more complex the computation and the net at that
                    moment. Low values are encouraged (Around 16 / 32)
            *) epochs
            *) loss
            *) optimizer
            *) BACKBONE (though this one is good, and low parameters)
            *) activation functions (all the time we use softmax): sigmoid, softmax, linear
            *) Amount of classes
                -) DONE: Amount of classes = 3 - Right and left lung, and infected mask
            *) Model type (Unet, FPN, Linknet, PSPNet), but U-Net is oriented for Medical Imaging
        '''
        # load your data
        x_train, y_train, x_val, y_val = input_images.get_training_data(training_ratio=0.8)

        # Dataset
        train_dataset = Dataset(x_train, y_train, classes=self.used_classes)
        validation_dataset = Dataset(x_val, y_val, classes=self.used_classes)

        train_dataloader = Dataloader(train_dataset, batch_size=self.batch_size, shuffle=True)
        valid_dataloader = Dataloader(validation_dataset, batch_size=1, shuffle=False)

        # check shapes for errors
        assert(train_dataloader[0][0].shape == (self.batch_size,) + input_images.image_size + (1,))
        assert(train_dataloader[0][1].shape == (self.batch_size,) + input_images.image_size + (len(self.used_classes),))

        # TODO: Implement this function properly
        self.create_and_train_model(self.used_classes, train_dataloader, valid_dataloader)

    def create_and_train_model(self, classes, training_loader, valid_loader):
        '''
        This function creates the custom model that will be used to segment.

        Args:
            backbone: Argument for Segmentation Models python module.
                There are 4 options for the model, but the backbone can be:
                    VGG	'vgg16' 'vgg19'
                    ResNet	'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'
                    SE-ResNet	'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'
                    ResNeXt	'resnext50' 'resnext101'
                    SE-ResNeXt	'seresnext50' 'seresnext101'
                    SENet154	'senet154'
                    DenseNet	'densenet121' 'densenet169' 'densenet201'
                    Inception	'inceptionv3' 'inceptionresnetv2'
                    MobileNet	'mobilenet' 'mobilenetv2'
                    EfficientNet	'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7'

            classes: List with strings that points out which are the classes that will be considered in the model.
                The list of options can be found in Database class, located at data_handlers module

            training_loader: DataLoader class with the training data
            valid_loader: DataLoader class with the validation data
        '''
        # First model introduced by Arsalan
        # Unet based model with the following characteristics
        # optimizer = Adam - Learning rate 0.0001
        # loss_func = Binary Cross-entropy
        # activation_func = softmax
        # It is based on core functions from Keras
        self.model = self.first_sketch_model(len(classes))

        # Unet model, found in GitHub
        # Characteristics
        # dropout = 10 % at each conv layer
        # batch normalization applied
        # activation_func = 'softmax'
        # optimizer = Adam - Learning rate 0.0001
        # loss_func = Categorical Cross-entropy
        # Model build based on basic Keras layers
        # self.model = self.unet_github(len(classes))

        # Complete Unet model based on Segmentation Models module
        # Characteristics
        # activation_func = 'sigmoid'
        # optimizer = Adam - Learning rate 0.0001
        # loss_func = BinaryCrossentropy()
        # backbone = 'efficientnetb7'
        # This class performs really well, with 99% accuracy
        # self.model = self.get_model_99_percent(len(classes))

        print(self.model.summary())

        callbacks = [
            EarlyStopping(monitor='val_loss', verbose=1, patience=10)
        ]

        self.model.fit(
            training_loader,
            validation_data=valid_loader,
            epochs = self.epochs,
            shuffle = self.shuffle,
            callbacks=callbacks,
            verbose = 2)

    def save_model(self, yaml_filepath, h5_weights_filepath):
        '''
        Save the generated model in a file in disk

        Args:
            filepath: Path where to store the model
        '''
        if not self.model is None:
            # serialize model to YAML
            model_yaml = self.model.to_yaml()
            with open(yaml_filepath, "w") as yaml_file:
                yaml_file.write(model_yaml)
            self.model.save_weights(h5_weights_filepath)
            print("Saved model to disk")
        else:
            print("Cannot save the model. It has not been created yet")

    def load_model(self, yaml_filepath, h5_weights_filepath):
        '''
        Load a model stored in the disk

        Args:
            filepath: Path where the model is
        Returns:
            True if the file exists and it could be opened
        '''
        ret = False

        if os.path.exists(yaml_filepath) and os.path.exists(h5_weights_filepath):
            # load YAML and create model
            with open(yaml_filepath, 'r') as yaml_file:
                loaded_model_yaml = yaml_file.read()

            self.model = model_from_yaml(loaded_model_yaml)
            # load weights into new model
            self.model.load_weights(h5_weights_filepath)
            print("Loaded model from disk")
            self.model.compile(optimizer = self.optimizer, loss = self.loss_func, metrics = ['accuracy'])

            ret = True
        else:
            print("In order to load a model, two files need to exists: the model and its weights. " + \
                "One of them or both are missing. Given paths:\nModel: {}\nWeights: {}".format(yaml_filepath, h5_weights_filepath))

        return ret

    def estimate_image(self, input):
        '''
        Estimate the mask of the input image

        Args:
            input: Graylevel input image.

        Returns:
            Image with the same size as the input, where each pixel is labeled
        '''
        # We resize to a known size, so the network does not crash
        resized_input = cv2.resize(input, self.resized_image_size, interpolation = cv2.INTER_AREA)

        # The model receives a batch of images with shape [N, Rows, Cols, Bands]. Since we have one
        # image and graylevel, N = Bands = 1
        predicted_mask = self.model.predict(resized_input.reshape((1,) + resized_input.shape + (1,)))[0]
        # For each pixel, we take the best prediction. The predicted mask has a shape
        # [N, Rows, Cols, 4]. The 4 is for each mask value. Each of them have the probabilities
        # that this pixels belongs to the label 0 (left lung), 1 (right lung), 2 (disease), or 3 (background).
        # Taking the highest probability, we are taking the label that is more likely to be
        out_image = np.array(np.argmax(predicted_mask, axis=2), dtype=np.uint8)

        # We resize the mask to the original image size
        out_image = cv2.resize(out_image, input.shape, interpolation = cv2.INTER_AREA)

        return out_image

    ####################################### MODEL DEFINITIONS ##############################################
    ## SEGMENTATION MODEL U-Net model
    def get_model_99_percent(self, classes):
        '''
        This function creates the custom model that will be used to segment.

        Args:
            backbone: Argument for Segmentation Models python module.
                There are 4 options for the model, but the backbone can be:
                    VGG	'vgg16' 'vgg19'
                    ResNet	'resnet18' 'resnet34' 'resnet50' 'resnet101' 'resnet152'
                    SE-ResNet	'seresnet18' 'seresnet34' 'seresnet50' 'seresnet101' 'seresnet152'
                    ResNeXt	'resnext50' 'resnext101'
                    SE-ResNeXt	'seresnext50' 'seresnext101'
                    SENet154	'senet154'
                    DenseNet	'densenet121' 'densenet169' 'densenet201'
                    Inception	'inceptionv3' 'inceptionresnetv2'
                    MobileNet	'mobilenet' 'mobilenetv2'
                    EfficientNet	'efficientnetb0' 'efficientnetb1' 'efficientnetb2' 'efficientnetb3' 'efficientnetb4' 'efficientnetb5' efficientnetb6' efficientnetb7'

            classes: List with strings that points out which are the classes that will be considered in the model.
                The list of options can be found in Database class, located at data_handlers module

            training_loader: DataLoader class with the training data
            valid_loader: DataLoader class with the validation data

            '''
        from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
        import segmentation_models as sm
        from keras.optimizers import Adam

        activation_func = 'sigmoid'
        # activation_func = 'softmax'
        optimizer = Adam(lr = 0.0001)
        loss_func = BinaryCrossentropy()
        # loss_func = CategoricalCrossentropy()
        backbone = 'efficientnetb7'

        base = sm.Unet(
            backbone,
            encoder_weights='imagenet',
            classes=classes,
            activation=activation_func,
            encoder_freeze=True,
            decoder_use_batchnorm=True)

        # We make the adaptation layer between grayscale images and RGB images
        inp = Input(shape=(None, None, 1))
        l1 = Conv2D(3, (1, 1))(inp)
        out = base(l1)
        model = Model(inp, out)
        model.compile(
            optimizer = optimizer,
            loss=loss_func,
            metrics = [sm.metrics.iou_score, 'accuracy'])

        return model

    ##### CUSTOM MODEL ######
    def put_layer(self, previous_layer, no_nuerons):
        '''
        This Function Adds layer in CNN model
        '''
        first_conv = Conv2D(no_nuerons, (3, 3), activation='relu', padding='same')(previous_layer)
        layer_complete = Conv2D(no_nuerons, (3, 3), activation='relu', padding='same')(first_conv)

        return layer_complete

    def inception_module(self, x,
                     filters_1x1,
                     filters_3x3_reduce,
                     filters_3x3,
                     filters_5x5_reduce,
                     filters_5x5,
                     filters_pool_proj,
                     name=None):

        from keras.initializers import glorot_uniform, Constant
        kernel_init = glorot_uniform()
        bias_init = Constant(value=0.2)

        conv_1x1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)

        conv_3x3 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        conv_3x3 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_3x3)

        conv_5x5 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(x)
        conv_5x5 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(conv_5x5)

        pool_proj = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        pool_proj = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', kernel_initializer=kernel_init, bias_initializer=bias_init)(pool_proj)

        output = concatenate([conv_1x1, conv_3x3, conv_5x5, pool_proj], axis=3, name=name)

        return output

    def first_sketch_model(self, classes):
        from keras.losses import BinaryCrossentropy, CategoricalCrossentropy
        from keras.optimizers import Adam

        optimizer = Adam(lr = 0.0001)
        loss_func = CategoricalCrossentropy()
        activation_func = 'softmax'

        inputs = Input((None, None, 1))

        layer_1 = self.inception_module(inputs, 64, 96, 128, 16, 32 , 32, name = 'inception_1')
        pool1 = MaxPooling2D(pool_size=(2, 2))(layer_1)
        drop = Dropout(0.3)(pool1)

        layer_2 = self.put_layer(drop,64)
        pool2 = MaxPooling2D(pool_size=(2, 2))(layer_2)

        layer_3 = self.put_layer(pool2,128)
        pool3 = MaxPooling2D(pool_size=(2, 2))(layer_3)

        layer_4 = self.put_layer(pool3, 256)
        pool4 = MaxPooling2D(pool_size=(2, 2))(layer_4)

        layer_5 = self.put_layer(pool4, 512)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(layer_5), layer_4], axis=3)
        layer_6 = self.put_layer(up6, 256)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(layer_6), layer_3], axis=3)
        layer_7 = self.put_layer(up7, 128)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(layer_7), layer_2], axis=3)
        layer_8 = self.put_layer(up8, 64)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(layer_8), layer_1], axis=3)
        layer_9 = self.put_layer(up9, 32)

        layer_10 = Conv2D(classes, (1, 1), activation=activation_func)(layer_9)

        model = Model(inputs=[inputs], outputs=[layer_10])
        model.compile(optimizer = optimizer, loss=loss_func, metrics = ['accuracy'])
        return model

    ### Model taken from GitHub
    # Link: https://gist.github.com/Laknath1996/6d9431379fc0d1a2d60f3b6982970978
    def conv2d_block(self, input_tensor, n_filters, kernel_size = 3, batchnorm = True):
        """Function to add 2 convolutional layers with the parameters passed to it"""
        # first layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                  kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # second layer
        x = Conv2D(filters = n_filters, kernel_size = (kernel_size, kernel_size),\
                  kernel_initializer = 'he_normal', padding = 'same')(input_tensor)
        if batchnorm:
            x = BatchNormalization()(x)
        x = Activation('relu')(x)

        return x

    def unet_github(self, classes):
        from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
        from keras.optimizers import Adam
        import segmentation_models as sm

        n_filters = 8               # Amount of filters for each layer
        dropout = 0.1               # How much dropout between layers?
        batchnorm = True            # Do we do batch normalization?
        activation_func = 'sigmoid' # activation function for the last layer
        # activation_func = 'softmax' # activation function for the last layer
        optimizer = Adam(lr = 0.0001)
        # loss_func = CategoricalCrossentropy()
        loss_func = BinaryCrossentropy()

        input_img = Input(shape=(None, None, 1))
        # Contracting Path
        c1 = self.conv2d_block(input_img, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)
        p1 = MaxPooling2D((2, 2))(c1)
        p1 = Dropout(dropout)(p1)

        c2 = self.conv2d_block(p1, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)
        p2 = MaxPooling2D((2, 2))(c2)
        p2 = Dropout(dropout)(p2)

        c3 = self.conv2d_block(p2, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)
        p3 = MaxPooling2D((2, 2))(c3)
        p3 = Dropout(dropout)(p3)

        c4 = self.conv2d_block(p3, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)
        p4 = MaxPooling2D((2, 2))(c4)
        p4 = Dropout(dropout)(p4)

        c5 = self.conv2d_block(p4, n_filters = n_filters * 16, kernel_size = 3, batchnorm = batchnorm)

        # Expansive Path
        u6 = Conv2DTranspose(n_filters * 8, (3, 3), strides = (2, 2), padding = 'same')(c5)
        u6 = concatenate([u6, c4])
        u6 = Dropout(dropout)(u6)
        c6 = self.conv2d_block(u6, n_filters * 8, kernel_size = 3, batchnorm = batchnorm)

        u7 = Conv2DTranspose(n_filters * 4, (3, 3), strides = (2, 2), padding = 'same')(c6)
        u7 = concatenate([u7, c3])
        u7 = Dropout(dropout)(u7)
        c7 = self.conv2d_block(u7, n_filters * 4, kernel_size = 3, batchnorm = batchnorm)

        u8 = Conv2DTranspose(n_filters * 2, (3, 3), strides = (2, 2), padding = 'same')(c7)
        u8 = concatenate([u8, c2])
        u8 = Dropout(dropout)(u8)
        c8 = self.conv2d_block(u8, n_filters * 2, kernel_size = 3, batchnorm = batchnorm)

        u9 = Conv2DTranspose(n_filters * 1, (3, 3), strides = (2, 2), padding = 'same')(c8)
        u9 = concatenate([u9, c1])
        u9 = Dropout(dropout)(u9)
        c9 = self.conv2d_block(u9, n_filters * 1, kernel_size = 3, batchnorm = batchnorm)

        outputs = Conv2D(classes, (1, 1), activation=activation_func)(c9)
        model = Model(inputs=[input_img], outputs=[outputs])
        model.compile(
            optimizer = optimizer,
            loss=loss_func,
            metrics = [sm.metrics.iou_score, 'accuracy'])
        return model