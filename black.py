#!/usr/bin/env python3
"""
make_black_midi.py

Download audio from a YouTube URL and convert it into a "Black MIDI" file by mapping every
spectral bin above a threshold to individual piano note events.

Features:
- Marks itself executable on first run
- Interactive prompts if URL or output path aren’t provided as arguments
- Uses yt-dlp for robust audio extraction

Usage:
    ./make_black_midi.py [YouTube_URL] [output.mid] [--threshold 0.02] [--hop 256] [--nfft 4096] [--res 960]

Dependencies:
    - yt-dlp CLI (on PATH)
    - ffmpeg CLI (on PATH)
    - Python packages: librosa, numpy, pretty_midi
"""
import os
import stat
import subprocess
import sys
import argparse
import librosa
import numpy as np
import pretty_midi

# Ensure the script is executable by the user
try:
    file_path = os.path.realpath(__file__)
    st = os.stat(file_path)
    os.chmod(file_path, st.st_mode | stat.S_IXUSR)
except Exception:
    pass


def download_audio(youtube_url: str, out_path: str = "audio.wav") -> str:
    """
    Downloads and extracts audio using yt-dlp, converting to WAV with ffmpeg.
    """
    # Use yt-dlp to extract best audio
    temp_audio = "temp_audio.%(ext)s"
    cmd = [
        "yt-dlp",
        "-x",                    # extract audio
        "--audio-format", "wav",
        "--output", temp_audio,
        youtube_url
    ]
    subprocess.run(cmd, check=True)
    # yt-dlp outputs e.g. temp_audio.wav
    wav_files = [f for f in os.listdir('.') if f.startswith('temp_audio') and f.endswith('.wav')]
    if not wav_files:
        raise RuntimeError("yt-dlp failed to produce a WAV file.")
    wav_file = wav_files[0]
    # Rename to standardized path
    os.replace(wav_file, out_path)
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
    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    pm = pretty_midi.PrettyMIDI(resolution=pm_resolution)
    instrument = pretty_midi.Instrument(program=0)

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
        description="Convert YouTube audio to a Black MIDI file using yt-dlp."
    )
    parser.add_argument('url', nargs='?', help='YouTube video URL')
    parser.add_argument('output', nargs='?', help='Path for the output MIDI file')
    parser.add_argument('--threshold', type=float, default=0.02,
                        help='Magnitude threshold for note creation')
    parser.add_argument('--hop', type=int, default=256,
                        help='STFT hop length in samples')
    parser.add_argument('--nfft', type=int, default=4096,
                        help='FFT window size')
    parser.add_argument('--res', type=int, default=960,
                        help='MIDI resolution (ticks per quarter note)')
    args = parser.parse_args()

    if not args.url:
        args.url = input("Enter YouTube video URL: ").strip()
    if not args.output:
        args.output = input("Enter output MIDI file path (e.g. output.mid): ").strip()

    print(f"Downloading and extracting audio from {args.url}...")
    try:
        audio_file = download_audio(args.url, out_path="temp_audio.wav")
    except subprocess.CalledProcessError as e:
        print("Error: Failed to download audio. Ensure yt-dlp and ffmpeg are installed and the URL is valid.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    print("Converting to Black MIDI...")
    try:
        audio_to_black_midi(
            audio_file,
            args.output,
            hop_length=args.hop,
            n_fft=args.nfft,
            magnitude_thresh=args.threshold,
            pm_resolution=args.res
        )
    except Exception as e:
        print(f"Error during MIDI conversion: {e}")
        sys.exit(1)

    os.remove(audio_file)
    print(f"Black MIDI file saved to {args.output}")

if __name__ == '__main__':
    main()
