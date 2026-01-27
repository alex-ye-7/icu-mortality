from data_processing import load_triplet_data
from triplet_dataset import TripletDataset, collate_triplets
import torch
from torch.utils.data import DataLoader

triplets_train, y_train, feature_to_id = load_triplet_data(
    "data/_processed", split_name="train"
)
print(triplets_train[0])

feature_to_id = torch.load("data/_processed/feature_to_id_train.pt")
print(feature_to_id)
