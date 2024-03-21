#!python3

# Code by Jioh L. Jung

from flask import Flask, request, jsonify
from faster_whisper import WhisperModel
import io
import base64
import soundfile as sf

import torch
import platform

app = Flask(__name__)

MODELS_LIST = [
    #"faster-whisper-large-v3",
    #"distil-small.en",
    "tiny.en",
]
MODELS = {}

DEFAULT_MODELS = "tiny.en"

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif platform.system() == "Darwin":  # macOS
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"

device = get_device()
print(f"> Using device: {device}")

for model_size in MODELS_LIST:
    try:
      MODELS[model_size] = WhisperModel(
        model_size,
        device=device,
        compute_type="float16" if device == "cuda" else "int8",
        )
      print('>> Loaded \'%s\' model with [devide:%s] Passed' % (model_size, device))
    except:
      MODELS[model_size] = WhisperModel(
        model_size,
        device="cpu",
        compute_type="float16" if device == "cuda" else "int8",
        )
      print('>> Loading \'%s\' model with [devide:%s] failed. using CPU. done.' % (model_size, device))

def get_models(model_name):
    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} not found")
    return MODELS[model_name]

def format_timestamp(seconds):
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f'{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f}'

@app.route('/v1/audio/transcriptions', methods=['POST'])
def transcribe():

    model_name = request.form.get('model', DEFAULT_MODELS)
    
    if model_name not in  MODELS_LIST:
        return jsonify({'error': 'unknown model name', 'model_list':MODELS_LIST}), 400
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    audio_file = request.files['file']

    if audio_file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    
    audio_bytes = io.BytesIO(audio_file.read())
    audio_data, sample_rate = sf.read(audio_bytes)
    audio_bytes = io.BytesIO()
    sf.write(audio_bytes, audio_data, sample_rate, format='WAV', subtype='PCM_16')
    audio_bytes.seek(0)

    min_silence_duration_ms = int(request.form.get('min_silence_duration_ms', 500))
    temperature = float(request.form.get('temperature', 0.0))
    language = request.form.get('language', 'auto')
    response_format = request.form.get('response_format', 'json')

    segments, info = get_models(model_name).transcribe(
        audio_bytes,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=min_silence_duration_ms),
        temperature=temperature,
        language=language,
    )

    if segments is None:
        return jsonify({'error': 'No speech detected'}), 400

    segments_result = [segment for segment in segments]

    if response_format == 'json':
        text = ' '.join(segment.text.strip() for segment in segments_result)
        return jsonify({'text': text}), 200
    elif response_format == 'text':
        text = ' '.join(segment.text.strip() for segment in segments_result)
        return text, 200
    elif response_format == 'srt':
        srt = ''
        for i, segment in enumerate(segments_result, start=1):
            srt += f'{i}\n'
            srt += f'{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n'
            srt += f'{segment.text}\n\n'
        return srt, 200, {'Content-Type': 'text/plain'}
    elif response_format == 'verbose_json':
        return jsonify({
            'task': 'transcribe',
            'language': language,
            'duration': info.duration,
            'text': ' '.join(segment.text for segment in segments_result),
            'segments': [{
                'id': i,
                'seek': 0,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text,
                'tokens': segment.tokens,
                'temperature': temperature,
                'avg_logprob': segment.avg_logprob,
                'compression_ratio': segment.compression_ratio,
                'no_speech_prob': segment.no_speech_prob
            } for i, segment in enumerate(segments_result)]
        }), 200
    elif response_format == 'vtt':
        vtt = 'WEBVTT\n\n'
        for segment in segments_result:
            vtt += f'{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n'
            vtt += f'{segment.text}\n\n'
        return vtt, 200, {'Content-Type': 'text/vtt'}
    else:
        return jsonify({'error': 'Invalid response format'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=2638, debug=False)