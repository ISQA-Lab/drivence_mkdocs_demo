from typing import List


class Label:
    def __init__(self, label_path: str = None) -> None:
        self.data = []

    def from_label(self, label: List) -> None:
        self.data = label

    def read_label(self, label_path: str):
        raise NotImplementedError("Label.read_label must be implemented by subclasses")

    def write_label(self, label_path: str):
        raise NotImplementedError("Label.write_label must be implemented by subclasses")

    def remove(self, ix):
        self.data.pop(ix)

    def insert(self, ix, obj):
        self.data.insert(ix, obj)

    def append(self, obj):
        self.data.append(obj)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        return self.data[ix]

    def get_all_boxes(self):
        raise NotImplementedError("Label.get_all_boxes must be implemented by subclasses")

    def get_care_labels(self):
        raise NotImplementedError("Label.get_care_labels must be implemented by subclasses")

    def get_dont_care_labels(self):
        raise NotImplementedError("Label.get_dont_care_labels must be implemented by subclasses")

    def organize_labels(self, inplace=True):
        raise NotImplementedError("Label.organize_labels must be implemented by subclasses")

    def sort_labels(self, inplace=True):
        raise NotImplementedError("Label.sort_labels must be implemented by subclasses")

