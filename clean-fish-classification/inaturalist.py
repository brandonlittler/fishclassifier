import re
import time
from pathlib import Path
import requests

# 
iNaturalistURL = "https://www.inaturalist.org/taxa/136345-Amphiprion-clarkii"


save_dir = Path("downloads")


download_limit = 100


licenses = "cc0,cc-by,cc-by-nc"


m = re.search(r"/taxa/(\d+)-", iNaturalistURL)
taxon_id = int(m.group(1)) if m else None
if taxon_id is None:
    raise ValueError("Could not extract ID from URL")

# --- Create a folder for this taxon ---
taxon_folder = save_dir / f"taxon_{taxon_id}"
taxon_folder.mkdir(parents=True, exist_ok=True)

# --- Query iNaturalist API for observations that have photos ---
api = "https://api.inaturalist.org/v1/observations"
per_page = 30
page = 1
downloaded = 0

session = requests.Session()
session.headers["User-Agent"] = "Simple-iNat-Image-Downloader/1.0"

while downloaded < download_limit:
    params = {
        "taxon_id": taxon_id,
        "photos": "true",
        "photo_license": licenses,
        "per_page": per_page,
        "page": page,
    }

    r = session.get(api, params=params, timeout=30)
    r.raise_for_status()
    results = r.json().get("results", [])

    if not results:  # no more data
        break

    for obs in results:
        obs_id = obs.get("id")
        for photo in obs.get("photos", []):
            if downloaded >= download_limit:
                break
            
            photo_id = photo.get("id")
            img_url = photo.get("url", "").replace("/square.", "/large.")
            
            if not img_url:
                continue
            
            # correct the naming convention to fit the others, and to be standard across all fish classes. 
            filename = save_dir / f"fish_{obs_id:012d}_{photo_id:05d}.png"
            
            if filename.exists():
                continue
            
            # download the image and save it to the filename 
            img = session.get(img_url, timeout=30)
            img.raise_for_status()
            filename.write_bytes(img.content)

            downloaded += 1
            print(f"Saved {filename}  <-  {img_url}")

    page += 1

print(f"\nDone. Downloaded {downloaded} images into: {taxon_folder}")

# move files from downloads/taxon_XXXX/ to fish_image/fish_XXXX/
source_dir = Path("downloads")
target_dir = Path("fish_image/fish_24")  # Or whatever fish class number
target_dir.mkdir(parents=True, exist_ok=True)

for png_file in source_dir.glob("fish_*.png"):
    target_path = target_dir / png_file.name
    png_file.rename(target_path)
    print (f"moved the file from {source_dir} to {target_dir} with filanem {target_path}")


     # clear up the downloads folder, to avoid duplicating files 

    downloads_dir = Path("downloads")
if downloads_dir.exists() and not any(downloads_dir.iterdir()):
    downloads_dir.rmdir()
    print(f"Deleted empty {downloads_dir}")