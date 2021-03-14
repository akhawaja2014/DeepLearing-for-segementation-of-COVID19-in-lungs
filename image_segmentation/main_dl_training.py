from file_management import FileManagement
from training_database import TrainingDataset
from dl_segmentation_model import DLSegmenationModel

import os

def main():
    '''
    Create and save a RandomForest Classifier model to segment the LV

    The location of the training dataset if harcoded in this function.
    It will automatically search for Nifty files that are located in that folder
    put them into a list, read the images, create the features, and then train
    the model with such data.
    '''
    dirname = os.path.dirname(os.path.abspath(__file__))
    print("Looking for files...")
    files_man = FileManagement(os.path.join(dirname, 'Database'))

    # Here it is ensure that we have the same amount of images than GT files
    pat_files = files_man.get_patients_filepaths()
    gt_files = files_man.get_gt_filepaths()

    print("Loading files...")
    train_set = TrainingDataset(storage_shape=(256,256))
    for i in range(len(pat_files)):
        train_set.append_data(pat_files[i], gt_files[i])

    import time
    ct = time.time()
    print("Training model")
    model = DLSegmenationModel(epochs=1, batch_size=4)
    model.train(train_set)
    model.save_model(
        os.path.join(dirname, 'deep_learning_model_{}.yaml'.format(int(ct))),
        os.path.join(dirname, 'deep_learning_model_{}.h5'.format(int(ct))))
    print("Elapsed time: {} minutes".format((time.time() - ct) / 60.0))

    print("Training finished successfully")

if __name__ == '__main__':
    main()