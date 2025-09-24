FROM pytorch/pytorch:2.1.0-cuda11.8-devel

WORKDIR /app

# Dipendenze di sistema
RUN apt-get update && apt-get install -y \
    git \
    libgl1 \
    libglib2.0-0 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copia requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clona GFPGAN
RUN git clone https://github.com/TencentARC/GFPGAN.git && \
    cd GFPGAN && \
    pip install -r requirements.txt && \
    python setup.py develop

# Clona Real-ESRGAN
RUN git clone https://github.com/xinntao/Real-ESRGAN.git && \
    cd Real-ESRGAN && \
    pip install -r requirements.txt && \
    python setup.py develop

# Scarica i modelli
# GFPGAN
RUN mkdir -p /root/.cache/gfpgan && \
    wget -O /root/.cache/gfpgan/GFPGANv1.4.pth \
    https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# RealESRGAN x2 (modello generico)
RUN mkdir -p /root/.cache/realesrgan && \
    wget -O /root/.cache/realesrgan/RealESRGAN_x2plus.pth \
    https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

# Copia codice
COPY app.py .
COPY static ./static

EXPOSE 7860

CMD ["python", "app.py"]
