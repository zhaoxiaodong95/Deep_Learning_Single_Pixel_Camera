import h5py
import torch.utils.data as data


class UCF101(data.Dataset):
    """
    Args:
        root (string): Root directory path.
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, hdf5_file):

        f = h5py.File(hdf5_file, 'r')
        self.data = f['training_64']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        clip = self.data[index]

        return clip

    def __len__(self):
        return len(self.data)


class UCF101_val(data.Dataset):
    """
    Args:
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, hdf5_file):

        f = h5py.File(hdf5_file, 'r')
        self.data = f['validation_64']

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        clip = self.data[index]

        return clip

    def __len__(self):
        return len(self.data)
