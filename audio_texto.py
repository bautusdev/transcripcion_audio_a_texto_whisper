import torch
from transformers import pipeline

# 1. Verificar si hay GPU disponible para acelerar el proceso
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Ejecutando en dispositivo: {device}")

# 2. Cargar el pipeline de reconocimiento automático de voz (ASR)
# Puedes usar 'openai/whisper-small', 'openai/whisper-base' o 'openai/whisper-large-v3'
model_id = "openai/whisper-small"

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=model_id,
    device=device,
    chunk_length_s=30,  # Procesa audios largos dividiéndolos en fragmentos de 30 segundos
    stride_length_s=5,
)

# 3. Definir parámetros específicos para forzar el idioma español
audio_path = "prueba.mp3"  # Cambia esto por tu archivo de audio (WAV, MP3, etc.)

transcripcion = asr_pipe(
    audio_path,
    generate_kwargs={"language": "spanish", "task": "transcribe"}
)

print("\n--- Texto Transcrito ---")
print(transcripcion["text"])
