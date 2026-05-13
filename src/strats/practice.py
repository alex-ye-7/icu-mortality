from strats.utils import load_triplet_data
from config import DATA_TRIPLET

triplets_list, y, feature_to_id = load_triplet_data(DATA_TRIPLET)

print(triplets_list[0])