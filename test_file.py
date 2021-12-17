from datasets.animal_dataset import AnimalDataset


ds = AnimalDataset()
print(len(ds))
print(ds.__getitem__(0).shape)
