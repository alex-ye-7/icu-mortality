import torch

# PyTorch Dataset and Collate Function
class TripletDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for triplet-based ICU data.
    Handles variable-length observation sequences.
    """
    
    def __init__(self, triplets_list, y, feature_to_id):

        self.triplets = triplets_list
        self.y = y
        self.feature_to_id = feature_to_id
        assert len(triplets_list) == len(y), "Mismatch between triplets and labels"
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        """
        Returns:
            triplets: Tensor of shape (num_obs, 3) - variable length
            label: Integer (0 or 1)
        """
        return self.triplets[idx], self.y[idx].long()


def collate_triplets(batch):
    """
    Collate for variable-length triplet sequences

    Params: batch List of (triplets, label) tuples from dataset
    Returns:
        padded_triplets tensor of (batch_size, max_seq_len, 3)
        labels: Tensor of shape (batch_size,)
        seq_lengths: Tensor of original sequence lengths for masking/RNN packing
    """
    triplets_list, labels = zip(*batch) # unpack and transpose
    labels = torch.stack(labels)
    seq_lengths = torch.tensor([len(t) for t in triplets_list], dtype=torch.long) # original lengths before pad

    padded_triplets = torch.nn.utils.rnn.pad_sequence( # pad to max length 
        triplets_list,
        batch_first=True,
        padding_value=0.0
    )
    
    return padded_triplets, labels, seq_lengths