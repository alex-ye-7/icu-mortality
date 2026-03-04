import argparse
import numpy as np
from tqdm import tqdm
from strats import STraTS
from dataset import TripletDataset

def parse_args() -> argparse.Namespace:
    """Function to parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--hid_dim', type=int, default=32)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attention_dropout', type=float, default=0.2)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    
    model = STraTS(args)
