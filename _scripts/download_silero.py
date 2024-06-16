import faster_whisper
import torch

device = "cpu"
torch.device(device)
model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',model='silero_vad',force_reload=True,onnx=False)
model.save('silero_vad.pth')

model = torch.jit.load('silero_vad.pth').eval().to(device)