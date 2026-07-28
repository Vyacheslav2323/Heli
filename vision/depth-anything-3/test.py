import os
import torch
from depth_anything_3.api import DepthAnything3
from depth_anything_3.services.input_handlers import VideoHandler

device = torch.device("cuda")
# DA3-SMALL is fastest (0.08B). Note: GS export needs DA3-GIANT or DA3NESTED-*.
model = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT")
model = model.to(device=device)

video_path = r"C:\Users\yj.park\Downloads\206294.mp4"
export_dir = "output"

# API expects a list of images, not a video file
images = VideoHandler.process(video_path, export_dir, fps=1.0)

prediction = model.inference(
    images,
    export_dir=export_dir,
    export_format="glb",  # Options: glb, npz, ply, mini_npz; gs_* needs GIANT/NESTED
)
