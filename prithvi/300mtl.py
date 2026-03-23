from huggingface_hub import snapshot_download

model_id_300 = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"
local_dir_300 = "./prithvi_300m_tl"

print("Downloading Prithvi-300M-TL weights...")
snapshot_download(repo_id=model_id_300, local_dir=local_dir_300)
print("Download complete.")