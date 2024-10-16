import gdown
import os

def download_file_from_google_drive(file_id,destination_folder):
    # Download a single file using its file ID
    url = f"https://drive.google.com/drive/folders/{file_id}"
    print(url)
    list_files = gdown.download_folder(url, output=destination_folder, quiet=False, use_cookies=False, remaining_ok=True, skip_download=True)
    print(list_files)
    for file in list_files:
        #print(file.id)
        url_id_file = f"https://drive.google.com/uc?id={file.id}&export=download&confirm=t"
        gdown.download(url = url_id_file, output = file.local_path, fuzzy = True, quiet=False, use_cookies=False)

# Replace 'file_ids' with the list of file IDs in the folder.
file_ids = [
    '1gmeE3D7R62UAtuIFOB9j2M5cUPTwtsxK',  # Replace with actual file ID
    '1n5w45suVOyaqY84hltJhIZdtVFD9B224', 
    '1mYOf5USMGNcJRPcvRVJVV1uHEalG5RPl',
    # Add more file IDs here...
]

destination_folder = './dataset/DOTA/'

# Make sure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Download each file
for file_id in file_ids:
    destination = os.path.join(destination_folder)  # Adjust file name and extension if needed
    download_file_from_google_drive(file_id, destination_folder)
    break

#  curl -L -o dataset/DOTA/labelTxt-v1.0.zip "https://drive.google.com/uc?export=download&id=1cCU7Mxs2YqQ26UBJcQklB1ZI6bv5fCt4"
#  unzip dataset/DOTA/labelTxt-v1.0.zip -d dataset/DOTA/labelTxt-v1.0/

# wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=1cCU7Mxs2YqQ26UBJcQklB1ZI6bv5fCt4' -O FILENAME