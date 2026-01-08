# Alexander Ye
# Load data from downloaded zip files, only need to run once

from zipfile import ZipFile

read_path = '/Users/alexye/Projects/ICUMortality/data/raw'
directory_to_extract_to = '/Users/alexye/Projects/ICUMortality/data/raw'
zip_file_paths = ["/set-a.zip", "/set-b.zip"]

for zip_file_path in zip_file_paths:
    print(f"Reading and extracting from {zip_file_path}")
    with ZipFile(read_path + zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(directory_to_extract_to)
    print(f"{zip_file_path} processed")