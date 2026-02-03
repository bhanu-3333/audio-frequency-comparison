import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
import matplotlib.pyplot as plt


@dataclass
class AudioFeatures:
    sample_rate: int
    mean_spectrum: np.ndarray
    freqs: np.ndarray
    rms: np.ndarray
    spectral_centroid: np.ndarray
    spectral_bandwidth: np.ndarray
    spectral_rolloff: np.ndarray
    spectral_flux: np.ndarray
    feature_vector: np.ndarray


def load_audio(path: Path, target_rate: int = 22050) -> Tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float32)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val
    if sample_rate != target_rate:
        data = resample_poly(data, target_rate, sample_rate)
        sample_rate = target_rate
    return sample_rate, data


def compute_features(sample_rate: int, data: np.ndarray) -> AudioFeatures:
    f, t, zxx = stft(data, fs=sample_rate, nperseg=1024, noverlap=512)
    magnitude = np.abs(zxx) + 1e-10
    mean_spectrum = magnitude.mean(axis=1)

    rms = np.sqrt(np.mean(magnitude**2, axis=0))
    spectral_centroid = (f[:, None] * magnitude).sum(axis=0) / magnitude.sum(axis=0)
    spectral_bandwidth = np.sqrt(
        ((f[:, None] - spectral_centroid) ** 2 * magnitude).sum(axis=0)
        / magnitude.sum(axis=0)
    )
    rolloff_threshold = 0.85 * magnitude.sum(axis=0)
    cumulative = np.cumsum(magnitude, axis=0)
    spectral_rolloff = np.array([
        f[np.searchsorted(cumulative[:, i], rolloff_threshold[i])] for i in range(cumulative.shape[1])
    ])

    spectral_flux = np.sqrt(np.sum(np.diff(magnitude, axis=1) ** 2, axis=0))

    feature_vector = np.array([
        mean_spectrum.mean(),
        mean_spectrum.std(),
        np.mean(rms),
        np.std(rms),
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.mean(spectral_rolloff),
        np.std(spectral_rolloff),
        np.mean(spectral_flux),
        np.std(spectral_flux),
    ])

    return AudioFeatures(
        sample_rate=sample_rate,
        mean_spectrum=mean_spectrum,
        freqs=f,
        rms=rms,
        spectral_centroid=spectral_centroid,
        spectral_bandwidth=spectral_bandwidth,
        spectral_rolloff=spectral_rolloff,
        spectral_flux=spectral_flux,
        feature_vector=feature_vector,
    )


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def spectrum_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = (a - a.mean()) / (a.std() + 1e-9)
    b_norm = (b - b.mean()) / (b.std() + 1e-9)
    return float(np.clip(np.corrcoef(a_norm, b_norm)[0, 1], -1.0, 1.0))


def dtw_distance(sequence_a: np.ndarray, sequence_b: np.ndarray) -> float:
    len_a, len_b = len(sequence_a), len(sequence_b)
    cost = np.full((len_a + 1, len_b + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            dist = abs(sequence_a[i - 1] - sequence_b[j - 1])
            cost[i, j] = dist + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(cost[len_a, len_b])


def compute_similarity(features_a: AudioFeatures, features_b: AudioFeatures) -> float:
    vector_similarity = cosine_similarity(features_a.feature_vector, features_b.feature_vector)
    spectrum_similarity = spectrum_correlation(features_a.mean_spectrum, features_b.mean_spectrum)
    dtw_dist = dtw_distance(features_a.spectral_centroid, features_b.spectral_centroid)
    dtw_similarity = 1.0 / (1.0 + dtw_dist)

    combined = 0.5 * vector_similarity + 0.3 * spectrum_similarity + 0.2 * dtw_similarity
    return float((combined + 1) / 2)


def plot_spectra(features_a: AudioFeatures, features_b: AudioFeatures, output_path: Path) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(features_a.freqs, 20 * np.log10(features_a.mean_spectrum + 1e-10), label="Audio 1")
    plt.plot(features_b.freqs, 20 * np.log10(features_b.mean_spectrum + 1e-10), label="Audio 2")
    plt.title("Mean Frequency Spectrum Comparison")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def analyze_audio(path_a: Path, path_b: Path, output_image: Path) -> float:
    sr_a, data_a = load_audio(path_a)
    sr_b, data_b = load_audio(path_b)
    if sr_a != sr_b:
        target_rate = min(sr_a, sr_b)
        sr_a, data_a = load_audio(path_a, target_rate=target_rate)
        sr_b, data_b = load_audio(path_b, target_rate=target_rate)

    features_a = compute_features(sr_a, data_a)
    features_b = compute_features(sr_b, data_b)
    similarity = compute_similarity(features_a, features_b)
    plot_spectra(features_a, features_b, output_image)
    return similarity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two audio recordings using frequency-domain analysis.")
    parser.add_argument("audio_a", type=Path, help="Path to the first WAV file.")
    parser.add_argument("audio_b", type=Path, help="Path to the second WAV file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("frequency_comparison.png"),
        help="Output path for the comparison plot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    similarity = analyze_audio(args.audio_a, args.audio_b, args.output)
    print(f"Similarity score: {similarity:.4f}")
    print(f"Saved frequency-domain comparison plot to {args.output}")


if __name__ == "__main__":
    main()
