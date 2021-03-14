# To search files in directories
import glob
import os

class FileManagement(object):
    '''
    Search Nifty folders in the given path, and split the filenames found
    on them into patient and masks data
    '''
    def __init__(self, filepath):
        self.database_filepath = os.path.abspath(filepath)
        if os.path.isfile(self.database_filepath):
            self.database_filepath = os.path.dirname(self.database_filepath)

        self.patient_folder_pattern = 'cases'
        self.ground_truth_folder_pattern = 'masks'
        self.nifty_file_pattern = '.nii.gz'

        self.patient_files = []
        self.gt_files = []

        self.search_nifty_files()

    def search_nifty_files(self):
        '''
        Look into the folder provided on construction if there are nifty files
        for both, the cases and the masks (GT). If a pair of those files is found,
        they will be stored in lists, that can be requested later.

        IMPORTANT: The database folder should consists of two folders with Nifty files: cases and masks.
        There can be other folders, but they should not contain nifty files.
        Each of them should have only Nifty files, and the corresponding mask should have
        the same name as the case file. Folders inside cases or masks will be ignored
        '''
        # Returns a list of names in list files.
        files = glob.glob(os.path.join(self.database_filepath, '**'), recursive = True)
        dirnames = []
        # Get directories that contains nifty files, without dupplications
        for file in files:
            if os.path.isfile(file) and self.nifty_file_pattern in file:
                directory = os.path.dirname(file)
                if not directory in dirnames:
                    dirnames.append(directory)

        # Sanity check: We only should have two folders in this place: Cases and masks
        if len(dirnames) == 0:
            print("The folder {d} is empty or it does not exists".format(d=self.database_filepath))
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

        print("{n} files found".format(n=len(self.patient_files)))

    def get_patients_filepaths(self):
        return self.patient_files

    def get_gt_filepaths(self):
        return self.gt_files

if __name__ == '__main__':
    dirname = os.path.dirname(os.path.abspath(__file__))
    files_man = FileManagement(os.path.join(dirname,'Database'))
    pat_files = files_man.get_patients_filepaths()
    gt_files = files_man.get_gt_filepaths()
    for i in range(len(pat_files)):
        print('----------------------------------------------')
        print("GT file")
        print(gt_files[i])
        print("Patient file")
        print(pat_files[i])
        print('')