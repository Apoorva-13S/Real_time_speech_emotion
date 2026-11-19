"""Live microphone emotion detection.

Improvements over the original live_emotion.py:
  * uses the project checkpoint by default, not a third-party model
  * picks the input device (the most common cause of "very quiet audio")
  * rolling block with an RMS gate instead of a hard 3 s record
  * normalises before inference (matches training) and reuses one model
  * prints latency, which the README listed as unmeasured
  * clean shutdown: the stream closes on Ctrl+C

Run:  python -m ser.live --model runs/local/best
      python -m ser.live --list-devices
"""

from __future__ import annotations

import os
import queue
import time

import numpy as np

from .predict import EmotionPredictor, format_prediction

BLOCK_S = 3.0
HOP_S = 1.5


def list_devices() -> int:
    try:
        import sounddevice as sd
    except Exception as exc:                             # noqa: BLE001
        print(f"sounddevice unavailable: {exc}\n"
              f"Linux: sudo apt-get install -y libportaudio2\n"
              f"macOS: brew install portaudio")
        return 1
    print(sd.query_devices())
    print(f"\ndefault input/output: {sd.default.device}")
    print("Set SER_MIC=<index or name> to choose an input device.")
    return 0


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Live microphone emotion detection")
    ap.add_argument("--model", default=None)
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--device", default=os.environ.get("SER_MIC"),
                    help="input device index/name (or set SER_MIC)")
    ap.add_argument("--block", type=float, default=BLOCK_S)
    ap.add_argument("--hop", type=float, default=HOP_S)
    ap.add_argument("--min-rms", type=float, default=0.005,
                    help="ignore blocks quieter than this")
    ap.add_argument("--top-k", type=int, default=3)
    args = ap.parse_args(argv)

    if args.list_devices:
        return list_devices()

    try:
        import sounddevice as sd
    except Exception as exc:                             # noqa: BLE001
        print(f"ERROR: sounddevice not usable: {exc}")
        print("Install PortAudio (see --list-devices) or use "
              "`python -m ser.predict --audio file.wav` instead.")
        return 1

    pred = EmotionPredictor(args.model)
    sr = pred.sr
    n_block = int(args.block * sr)
    n_hop = max(1, int(args.hop * sr))

    dev = args.device
    if dev is not None and str(dev).isdigit():
        dev = int(dev)
    print(f"\nlistening on device={dev if dev is not None else 'system default'}"
          f"  block={args.block}s  hop={args.hop}s")
    print("Speak normally. Press Ctrl+C to stop.\n")

    q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"  (stream status: {status})")
        q.put(indata.copy())

    buf = np.zeros(0, dtype=np.float32)
    try:
        with sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                            blocksize=max(sr // 10, 256), device=dev,
                            callback=callback):
            last = 0.0
            while True:
                chunk = q.get()
                buf = np.concatenate([buf, chunk.ravel().astype(np.float32)])
                if len(buf) < n_block:
                    continue
                window = buf[-n_block:]
                if time.time() - last < args.hop:
                    continue
                rms = float(np.sqrt(np.mean(np.square(window)) + 1e-12))
                if rms < args.min_rms:
                    print("  . (too quiet -- check the input device with "
                          "--list-devices)", flush=True)
                    last = time.time()
                    continue
                t0 = time.perf_counter()
                preds = pred.predict(window, sr, top_k=args.top_k)
                dt = (time.perf_counter() - t0) * 1000
                print(f"{format_prediction(preds, args.top_k)}"
                      f"   [{dt:.0f} ms]", flush=True)
                last = time.time()
                buf = buf[-n_hop:]
    except KeyboardInterrupt:
        print("\nstopped.")
    except Exception as exc:                             # noqa: BLE001
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        print("If this is a PortAudio error, run `python -m ser.live "
              "--list-devices` and pick a real input device with --device.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
