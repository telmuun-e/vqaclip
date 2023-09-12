import requests

URL = "https://onedrive.live.com/download?resid=FD39CA9D49B27CFF%211499&authkey=!ADfN10pnPuhHUQs"
MODEL_PATH = "model/best_model.pt"

print("Downloading the model...")
r = requests.get(URL)
open(MODEL_PATH, "wb").write(r.content)
