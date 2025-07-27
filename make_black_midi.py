#!/usr/bin/env python3
"""
make_black_midi.py

Download audio from a YouTube URL and convert it into a "Black MIDI" file by mapping every
spectral bin above a threshold to individual piano note events.

Usage:
    python make_black_midi.py <YouTube_URL> <output.mid> [--threshold 0.02] [--hop 256] [--nfft 4096] [--res 960]
"""
import argparse
import os
import subprocess
from pytube import YouTube
import librosa
import numpy as np
import pretty_midi

def download_audio(youtube_url: str, out_path: str = "audio.wav") -> str:
    """
    Downloads the audio stream from the given YouTube URL and converts it to a WAV file.
    """
    yt = YouTube(youtube_url)
    stream = yt.streams.filter(only_audio=True).first()
    temp_mp4 = stream.download(filename="temp_audio.mp4")
    subprocess.run([
        "ffmpeg", "-y", "-i", temp_mp4,
        "-ar", "22050", "-ac", "1", out_path
    ], check=True)
    os.remove(temp_mp4)
    return out_path


def audio_to_black_midi(
    audio_path: str,
    output_midi: str,
    hop_length: int = 256,
    n_fft: int = 4096,
    magnitude_thresh: float = 0.02,
    pm_resolution: int = 960
) -> None:
    """
    Converts a WAV audio file into a Black MIDI by creating a MIDI note for every time-frequency bin
    whose magnitude exceeds the specified threshold.
    """
    # Load and compute spectrogram
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Prepare PrettyMIDI object
    pm = pretty_midi.PrettyMIDI(resolution=pm_resolution)
    instrument = pretty_midi.Instrument(program=0)

    # Map each spectral bin above threshold to a note
    for t_idx, t in enumerate(times):
        mags = S[:, t_idx]
        high_idxs = np.where(mags > magnitude_thresh)[0]
        for i in high_idxs:
            f = freqs[i]
            note_number = int(round(pretty_midi.hz_to_note_number(f)))
            if 0 <= note_number <= 127:
                velocity = int(np.clip(mags[i] * 127, 1, 127))
                note = pretty_midi.Note(
                    velocity=velocity,
                    pitch=note_number,
                    start=t,
                    end=t + (hop_length / sr)
                )
                instrument.notes.append(note)

    pm.instruments.append(instrument)
    pm.write(output_midi)


def main():
    parser = argparse.ArgumentParser(
        description="Convert YouTube audio to a Black MIDI file."
    )
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('output', help='Path for the output MIDI file')
    parser.add_argument('--threshold', type=float, default=0.02,
                        help='Magnitude threshold for note creation')
    parser.add_argument('--hop', type=int, default=256,
                        help='STFT hop length in samples')
    parser.add_argument('--nfft', type=int, default=4096,
                        help='FFT window size')
    parser.add_argument('--res', type=int, default=960,
                        help='MIDI resolution (ticks per quarter note)')
    args = parser.parse_args()

    print(f"Downloading audio from {args.url}...")
    audio_file = download_audio(args.url, out_path="temp_audio.wav")

    print("Converting to Black MIDI...")
    audio_to_black_midi(
        audio_file,
        args.output,
        hop_length=args.hop,
        n_fft=args.nfft,
        magnitude_thresh=args.threshold,
        pm_resolution=args.res
    )

    os.remove(audio_file)
    print(f"Black MIDI file saved to {args.output}")

if __name__ == '__main__':
    main()
