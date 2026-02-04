# Audio Frequency Comparison and Voice Similarity Evaluation

---

This Python-based application compares two audio recordings and evaluates their similarity using frequency-domain analysis. Instead of relying solely on pitch, the system examines frequency distribution, intensity variations, and time-based behavior of audio signals to generate a numerical similarity score.

The application is implemented as a Flask web service that accepts two audio files as input and returns a similarity evaluation.

---

## Requirements

---

- Python 3  
- Required Python libraries (listed in `requirements.txt`)

---

## Running the Application

---

```bash
python app.py
The service will be available at:

http://localhost:5000
Audio Comparison API
Endpoint: /match/CompareAudio
Method: POST

Example Request
curl -X POST \
-F "ref_audio_name=reference_audio" \
-F "q_audio=@/path/to/query_audio.wav" \
-F "uid=123" \
http://localhost:5000/match/CompareAudio
The response includes a similarity label and a similarity score indicating how closely the two audio recordings match.