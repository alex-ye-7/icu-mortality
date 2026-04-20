# Alexander Ye
# Dataset and Dataloader for triplet form

import torch
import numpy as np
from pathlib import Path
from triplet.utils.io import *
from triplet.utils.triplets_utils import *

class Dataset:
    def __init__(self, split='train', stat_indices=None) -> None:
        data_dir = Path(__file__).parent.parent / 'data' / '_processed'
        triplets_list, y, feature_to_id = load_triplet_data(str(data_dir), split_name=split)

        # Extract demographics -> make this better
        self.demo = normalize_demographics(triplets_list, feature_to_id)

        # Z-score normalize time-series values
        # Compute stats on train subset only, reuse for test
        if split == 'train':
            stat_subset = [triplets_list[i] for i in stat_indices] if stat_indices is not None else triplets_list
            stats = compute_value_stats(stat_subset)
            save_value_stats(stats, str(data_dir))
        else:
            stats = load_value_stats(str(data_dir))
        normalize_triplet_values(triplets_list, stats)

        # Convert triplets to separate arrays of times, feature IDs, and values
        self.times, self.values, self.varis  = [], [], []
        self.y = y.numpy()
        
        for triplet in triplets_list:
            if len(triplet) > 0:
                # Convert to tensor if needed
                if not isinstance(triplet, torch.Tensor):
                    triplet = torch.tensor(triplet, dtype=torch.float32)
                    
                times = triplet[:, 0].numpy().tolist()
                var_ids = triplet[:, 1].numpy().astype(int).tolist()
                values = triplet[:, 2].numpy().tolist()
            else:
                times, var_ids, values = [], [], []
                
            self.times.append(times)
            self.varis.append(var_ids)
            self.values.append(values)
        
        self.n_samples = len(self.values)

    def __len__(self):
        return self.n_samples
    
    def get_batch(self, ind):
        """ Create a padded batch from sample indices.
        
        Args: ind: List or array of sample indices
        Returns: Dictionary with batched tensors
        """
        demo = torch.FloatTensor(self.demo[ind]) # (batch_size, D)
        num_obs = [len(self.values[i]) for i in ind]
        max_obs = max(num_obs) if num_obs else 1
        pad_lens = max_obs - np.array(num_obs)
        
        values = [self.values[i] + [0.0]*(l) for i, l in zip(ind, pad_lens)]
        times = [self.times[i] + [0.0]*(l) for i, l in zip(ind, pad_lens)]
        varis = [self.varis[i] + [0]*(l) for i, l in zip(ind, pad_lens)]
        
        values = torch.FloatTensor(values)
        times = torch.FloatTensor(times)
        varis = torch.LongTensor(varis)
        
        obs_mask = [[1]*l1 + [0]*l2 for l1, l2 in zip(num_obs, pad_lens)]
        obs_mask = torch.FloatTensor(obs_mask)
        
        return {
            'values': values,
            'times': times,
            'varis': varis,
            'obs_mask': obs_mask,
            'demo': demo,
            'labels': torch.FloatTensor(self.y[ind])
        }


class DataLoader:
    """Custom DataLoader for handling variable-length sequences in batches."""
    def __init__(self, dataset, batch_size=32, shuffle=True, indices=None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = indices if indices is not None else np.arange(len(dataset))

    def __iter__(self):
        indices = self.indices.copy()
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = min(start_idx + self.batch_size, len(indices))
            batch_indices = indices[start_idx:end_idx]
            yield self.dataset.get_batch(batch_indices) # memory efficient

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))