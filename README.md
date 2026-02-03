# Audio Similarity Analyzer

This project provides a frequency-domain analysis pipeline to compare two audio recordings and compute a similarity score based on spectral patterns, intensity variations, and time-based frequency behavior.

## Features
- Frequency distribution patterns via mean spectrum.
- Intensity (amplitude) variations via RMS energy stats.
- Time-based frequency behavior via spectral centroid, bandwidth, rolloff, and flux.
- Quantitative similarity score combining spectral correlation, feature-vector similarity, and DTW alignment.

## Usage

```bash
python audio_similarity.py path/to/audio1.wav path/to/audio2.wav --output comparison.png
```

The script prints a similarity score (0 to 1) and saves a combined frequency-domain plot.

## Output
- A single frequency-domain visualization showing both recordings' mean spectra.
- A numerical similarity score printed to stdout.
