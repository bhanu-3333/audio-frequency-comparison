🎧 Audio Frequency Comparison and Voice Similarity Evaluation

This Python-based application compares two audio recordings and evaluates their similarity using frequency-domain analysis. Instead of relying only on pitch, the system analyzes frequency distribution, intensity variations, and time-based behavior of audio signals to produce a numerical similarity score.

The application runs as a Flask web service and accepts two audio files as input for comparison.

⚙️ Requirements

Python 3

Required Python libraries (requirements.txt)

▶️ Running the Application
python app.py


The service will be available at:

http://localhost:5000

🔁 Audio Comparison API

Endpoint: /match/CompareAudio
Method: POST

Example:
curl -X POST \
-F "ref_audio_name=reference_audio" \
-F "q_audio=@/path/to/query_audio.wav" \
-F "uid=123" \
http://localhost:5000/match/CompareAudio


The response includes a similarity label and a similarity score indicating how closely the two audio recordings match.

✨ Key Features

Frequency-domain analysis using STFT and CQT

Chroma, pitch, and spectral feature extraction

Similarity evaluation using Qmax, Dmax, and related measures

Scoring for consistency, pronunciation, intonation, pitch, and speed