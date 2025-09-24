# Costruisci
docker build -t upscaler-web .

# Esegui con GPU
docker run --gpus all -p 7860:7860 upscaler-web

Modello	Migliore per	Tipo di upscaling
GFPGAN	Volti umani	Ripristino + upscaling
Real-ESRGAN	Tutto (paesaggi, anime, oggetti)	Upscaling generico
