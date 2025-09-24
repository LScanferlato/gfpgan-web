from flask import Flask, request, send_file, render_template_string
from PIL import Image
import numpy as np
import io
import torch

app = Flask(__name__)

# --- Inizializzazione modelli ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GFPGAN (per volti)
import sys
sys.path.append('/app/GFPGAN')
from gfpgan import GFPGANer
gfpgan = GFPGANer(
    model_path='/root/.cache/gfpgan/GFPGANv1.4.pth',
    upscale=2,
    arch='clean',
    channel_multiplier=2,
    bg_upsampler=None,
    device=device
)

# Real-ESRGAN (per generico)
sys.path.append('/app/Real-ESRGAN')
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# Modello: RealESRGAN x2+
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
realesrgan = RealESRGANer(
    scale=2,
    model_path='/root/.cache/realesrgan/RealESRGAN_x2plus.pth',
    model=model,
    tile=400,
    tile_pad=10,
    pre_pad=0,
    half=True,  # migliora velocità su GPU
    device=device
)

@app.route('/')
def index():
    with open('static/index.html', 'r') as f:
        html = f.read()
    return render_template_string(html)

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "Nessun file", 400

    file = request.files['file']
    model_choice = request.form.get('model', 'gfpgan')  # 'gfpgan' o 'realesrgan'

    try:
        input_img = Image.open(file.stream)
    except Exception as e:
        return f"Errore lettura immagine: {e}", 400

    # 1. Riduci del 50%
    w, h = input_img.size
    low_res = input_img.resize((w // 2, h // 2), Image.LANCZOS)

    # 2. Converti in numpy
    low_res_np = np.array(low_res)
    if len(low_res_np.shape) == 2:
        low_res_np = np.tile(low_res_np[:, :, None], 3)
    if low_res_np.shape[2] == 4:
        low_res_np = low_res_np[:, :, :3]

    # 3. Elabora con modello scelto
    if model_choice == 'realesrgan':
        output_np = realesrgan.enhance(low_res_np)[0]  # [0] = img, [1] = tensor
    else:  # gfpgan
        _, _, output_np = gfpgan.enhance(low_res_np, has_aligned=False)

    # 4. Converti in PIL e invia
    output_pil = Image.fromarray(output_np)

    img_io = io.BytesIO()
    output_pil.save(img_io, 'PNG')
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)
