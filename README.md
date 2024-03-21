# Whisper API Server (OpenAI compatible API)
This is a Flask-based API server that provides speech-to-text transcription functionality using the Faster Whisper library. It supports multiple pre-trained models and allows users to transcribe audio files in various formats.

This server's API is fully compatible original OpenAI's whisper API spec.
- https://platform.openai.com/docs/api-reference/audio/json-object

# Author
- code by Jioh L. Jung
- Code and manual readme file was created with Claude3 opus(AI assisted).

# License
- MIT License

# Requirements
- Python 3.6 or higher
- Flask
- faster-whisper
- soundfile
- torch

# Installation
- Clone the repository or download the source code.
- Install the required dependencies using pip:

```
pip install -r requirements.txt
```

- Download the pre-trained models you want to use and place them in the appropriate directory.

# Usage
Start the API server by running the following command:

```
python api-server.py
```
- The server will start running on http://0.0.0.0:2638.

- Send a POST request to the /v1/audio/transcriptions endpoint with the following parameters:
 - file (required): The audio file to be transcribed. Supported formats: WAV, MP3, MP4, MPEG, MPGA, M4A, WEBM.
 - model (optional): The name of the pre-trained model to use for transcription. Default: tiny.en. Available models are defined in the MODELS_LIST variable.
 - min_silence_duration_ms (optional): The minimum duration of silence (in milliseconds) to be considered as a separate segment. Default: 500.
 - temperature (optional): The temperature value for the transcription model. Default: 0.0.
 - language (optional): The language of the audio file. Default: auto (automatic detection).
 - response_format (optional): The format of the response. Supported formats: json, text, srt, verbose_json, vtt. Default: json.

- The API will respond with the transcription result in the specified format.

# Transcription Process
- The API server receives the audio file and reads it into memory using io.BytesIO().
- The audio data is then converted to WAV format using the soundfile library.
- The Faster Whisper model is loaded based on the specified model parameter. The server attempts to use the available device in the following order: CUDA, MPS (for macOS), CPU.
- The audio data is transcribed using the loaded model with the specified parameters (min_silence_duration_ms, temperature, language).
- The transcription result is formatted according to the response_format parameter and returned as the API response.

# Supported Response Formats
- json: Returns the transcription result as a JSON object with a single text field containing the transcribed text.
- text: Returns the transcribed text as plain text.
- srt: Returns the transcription result in the SubRip (SRT) subtitle format.
- verbose_json: Returns a detailed JSON object containing additional information about the transcription, such as segment timestamps, tokens, and probabilities.
- vtt: Returns the transcription result in the Web Video Text Tracks (WebVTT) format.

# Error Handling
The API server handles the following error scenarios:

- If no file is uploaded, it returns a 400 Bad Request response with an error message.
- If an unknown model name is specified, it returns a 400 Bad Request response with an error message and the list of available models.
- If no speech is detected in the audio file, it returns a 400 Bad Request response with an error message.
Notes
- The server runs on http://0.0.0.0:2638 by default. You can modify the host and port parameters in the app.run() function call to change the server's address and port.
- The available pre-trained models are defined in the MODELS_LIST variable. You can uncomment or add more models as needed.
- The server attempts to use the available device for model loading in the following order: CUDA, MPS (for macOS), CPU. If a device is not available or fails, it falls back to the next available device.
