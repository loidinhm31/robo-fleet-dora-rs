#!/usr/bin/env python3
"""
USB Camera Microphone Checker
=============================
Records >= 10 seconds from the USB camera microphone, analyses the captured
audio (level, silence ratio, clipping, DC offset, SNR estimate), and tries to
play it back so the full mic -> speaker loop can be verified.

Run it, then copy everything between the markers back to me:

    python3 scripts/check-usb-microphone.py

Options:
    --duration N        seconds to record (default 10, must be >= 1)
    --device DEVICE     ALSA capture device, e.g. "hw:0,0" (default: auto-detect USB cam)
    --rate N            sample rate (default 48000)
    --channels N        1=mono 2=stereo (default 2)
    --output PATH       where to save the wav (default /tmp/usb_mic_check.wav)
    --no-playback       skip the playback test
    --pipewire-sink N   PipeWire sink node id for playback (default: auto)

Only the Python standard library is used (no numpy / no pyaudio needed).
"""
import argparse
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import wave

REPORT_MARKER_BEGIN = "=====BEGIN USB MIC REPORT====="
REPORT_MARKER_END = "=====END USB MIC REPORT====="


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(msg, flush=True)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}", file=sys.stderr, flush=True)


def run(cmd, timeout=None):
    """Run a command, return (returncode, stdout, stderr)."""
    try:
        p = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout after {timeout}s: {' '.join(cmd)}"


def have(cmd: str) -> bool:
    return shutil.which(cmd) is not None


# --------------------------------------------------------------------------- #
# Device detection
# --------------------------------------------------------------------------- #
def detect_usb_mic():
    """Parse `arecord -l` and return (alsa_device, card_num, description) for the
    first USB Audio capture device, preferring one named 'Camera'."""
    rc, out, err = run(["arecord", "-l"])
    if rc != 0:
        return None, None, f"arecord -l failed: {err.strip() or out.strip()}"
    cards = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("card "):
            continue
        # card 0: Camera [PC-LM1E Camera], device 0: USB Audio [USB Audio]
        try:
            card_num = int(line.split(":")[0].split()[1])
            rest = line.split(",", 1)[0]  # card 0: Camera [PC-LM1E Camera]
            name = rest.split("[", 1)[1].split("]", 1)[0] if "[" in rest else f"card{card_num}"
            is_usb = "USB Audio" in line
            cards.append((card_num, name, line, is_usb))
        except Exception:
            continue
    usb = [c for c in cards if c[3]]
    if not usb:
        return None, None, "No USB Audio capture device found in `arecord -l`."
    # Prefer a card whose description/name mentions Camera
    cam = [c for c in usb if "camera" in (c[1] + " " + c[2]).lower()]
    chosen = cam[0] if cam else usb[0]
    card_num, name, desc, _ = chosen
    return f"hw:{card_num},0", card_num, desc


# --------------------------------------------------------------------------- #
# Recording
# --------------------------------------------------------------------------- #
def record(device, duration, rate, channels, out_path):
    """Record `duration` seconds with a live countdown. Returns (ok, msg)."""
    cmd = [
        "arecord", "-D", device,
        "-f", "S16_LE", "-r", str(rate), "-c", str(channels),
        "-d", str(duration), "-q",
        out_path,
    ]
    log(f"\n[REC] Starting {duration}s recording from {device} ...")
    log("[REC] >>> MAKE SOME NOISE NOW (talk / clap / play music) <<<")
    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
    )
    start = time.monotonic()
    try:
        while proc.poll() is None:
            elapsed = time.monotonic() - start
            remaining = max(0, duration - elapsed)
            sys.stdout.write(
                f"\r[REC] Recording... {elapsed:4.1f}s / {duration}s "
                f"(remaining {remaining:4.1f}s)   "
            )
            sys.stdout.flush()
            if elapsed >= duration + 2:
                break
            time.sleep(0.1)
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
    finally:
        sys.stdout.write("\r" + " " * 70 + "\r")
        sys.stdout.flush()
    stderr = (proc.stderr.read() if proc.stderr else "") or ""
    if proc.returncode != 0:
        return False, f"arecord exited {proc.returncode}: {stderr.strip()}"
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 44:
        return False, f"recording file missing/too small: {out_path} ({stderr.strip()})"
    return True, stderr.strip()


# --------------------------------------------------------------------------- #
# Audio analysis (stdlib only)
# --------------------------------------------------------------------------- #
def analyse(path):
    """Return a dict of statistics about the wav file."""
    w = wave.open(path, "rb")
    rate = w.getframerate()
    channels = w.getnchannels()
    sampwidth = w.getsampwidth()
    nframes = w.getnframes()
    duration = nframes / rate if rate else 0.0
    raw = w.readframes(nframes)
    w.close()

    if sampwidth != 2:
        return {"error": f"unsupported sample width {sampwidth} bytes (expected 16-bit)"}

    fmt = "<" + str(len(raw) // 2) + "h"
    try:
        samples = struct.unpack(fmt, raw)
    except struct.error as e:
        return {"error": f"could not decode PCM: {e}"}

    total = len(samples)
    if total == 0:
        return {"error": "no samples in file"}

    peak = 0
    sum_sq = 0.0
    sum_val = 0.0
    clipped = 0
    silent = 0
    SILENCE_THRESH = 300       # ~-40 dBFS for 16-bit
    CLIP_THRESH = 32767

    for s in samples:
        a = abs(s)
        if a > peak:
            peak = a
        sum_sq += s * s
        sum_val += s
        if a >= CLIP_THRESH:
            clipped += 1
        if a < SILENCE_THRESH:
            silent += 1

    rms = math.sqrt(sum_sq / total)
    dc_offset = sum_val / total
    rms_dbfs = 20 * math.log10(rms / 32768) if rms > 0 else -999.0
    peak_dbfs = 20 * math.log10(peak / 32768) if peak > 0 else -999.0
    silence_pct = 100.0 * silent / total
    clip_pct = 100.0 * clipped / total

    # crude SNR estimate: assume the noise floor is the silent-period RMS
    silent_rms_sq = 0.0
    silent_n = 0
    for s in samples:
        if abs(s) < SILENCE_THRESH:
            silent_rms_sq += s * s
            silent_n += 1
    noise_rms = math.sqrt(silent_rms_sq / silent_n) if silent_n else 1.0
    snr_db = 20 * math.log10(rms / noise_rms) if rms > 0 and noise_rms > 0 else 0.0

    # verdict
    if peak < SILENCE_THRESH:
        verdict = "SILENT (no usable signal)"
    elif peak < 1500:
        verdict = "VERY LOW (barely audible)"
    elif rms_dbfs < -45:
        verdict = "LOW (signal present but quiet)"
    elif clip_pct > 1.0:
        verdict = "CLIPPING (too loud / gain too high)"
    elif rms_dbfs > -12:
        verdict = "LOUD (healthy level)"
    else:
        verdict = "OK (usable signal)"

    return {
        "channels": channels,
        "rate": rate,
        "sampwidth": sampwidth,
        "nframes": nframes,
        "duration": duration,
        "total_samples": total,
        "peak": peak,
        "peak_dbfs": peak_dbfs,
        "rms": rms,
        "rms_dbfs": rms_dbfs,
        "dc_offset": dc_offset,
        "silence_pct": silence_pct,
        "clip_pct": clip_pct,
        "snr_db": snr_db,
        "verdict": verdict,
    }


# --------------------------------------------------------------------------- #
# Playback
# --------------------------------------------------------------------------- #
def setup_pipewire_env():
    """Populate XDG_RUNTIME_DIR / DBUS so PipeWire clients can connect."""
    uid = os.getuid()
    env = dict(os.environ)
    if not env.get("XDG_RUNTIME_DIR") and os.path.isdir(f"/run/user/{uid}"):
        env["XDG_RUNTIME_DIR"] = f"/run/user/{uid}"
    if not env.get("DBUS_SESSION_BUS_ADDRESS"):
        bus = f"/run/user/{uid}/bus"
        if os.path.exists(bus):
            env["DBUS_SESSION_BUS_ADDRESS"] = f"unix:path={bus}"
    return env


def find_default_sink():
    """Return PipeWire sink node id (str) for the analog output, or None."""
    rc, out, _ = run(["wpctl", "status"])
    if rc != 0:
        return None
    in_sinks = False
    for line in out.splitlines():
        s = line.strip()
        if s.startswith("Sinks"):
            in_sinks = True
            continue
        if in_sinks and s.startswith("Sources"):
            break
        if in_sinks and "*" in line and "Analog Stereo" in line:
            # *   55. Ryzen HD Audio Controller Analog Stereo [vol: 1.00]
            try:
                return line.split(".", 1)[0].replace("*", "").strip()
            except Exception:
                continue
    return None


def playback(path, pipewire_sink, no_playback):
    """Try to play the file. Returns (method, exit_code, message)."""
    if no_playback:
        return "skipped", 0, "--no-playback given"

    pw_msg = ""
    if have("pw-play"):
        sink = pipewire_sink or find_default_sink()
        cmd = ["pw-play"]
        if sink:
            cmd += ["--target", sink]
        cmd += [path]
        rc, out, err = run(cmd, timeout=60)
        if rc == 0:
            return "pw-play", rc, f"played via PipeWire sink {sink or '(default)'}"
        pw_msg = f"pw-play exit {rc}: {(err or out).strip()}"

    if have("aplay"):
        rc, out, err = run(["aplay", "-q", path], timeout=60)
        if rc == 0:
            return "aplay(default)", rc, "played via aplay default device"
        # try to auto pick an analog playback device
        rc2, lout, _ = run(["aplay", "-l"])
        if rc2 == 0:
            for line in lout.splitlines():
                if "ALC" in line or "Analog" in line:
                    try:
                        card = line.split(":")[0].split()[1]
                        rc3, o3, e3 = run(
                            ["aplay", "-q", "-D", f"plughw:{card},0", path], timeout=60
                        )
                        if rc3 == 0:
                            return f"aplay(plughw:{card},0)", rc3, "played via analog card"
                    except Exception:
                        pass
        return "aplay", rc, f"aplay exit {rc}: {(err or out).strip()}" + (
            f" | pw-play: {pw_msg}" if pw_msg else "")

    return "none", 1, "no playback tool found (need pw-play or aplay)"


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def print_report(args, device, card_num, dev_desc, rec_ok, rec_msg,
                 stats, play_method, play_rc, play_msg, tools):
    lines = []
    a = lines.append

    a(REPORT_MARKER_BEGIN)
    a("# USB Camera Microphone Check")
    a("")
    a("## Environment / tools")
    a(f"- date: {time.strftime('%Y-%m-%d %H:%M:%S %z')}")
    a(f"- python: {sys.version.split()[0]}")
    a(f"- tools found: {', '.join(tools) if tools else 'none'}")
    a("")
    a("## Device")
    a(f"- arecord device: {device or 'NOT FOUND'}")
    a(f"- card number: {card_num if card_num is not None else 'n/a'}")
    a(f"- description: {dev_desc}")
    a("")
    a("## Recording settings")
    a(f"- duration: {args.duration}s")
    a(f"- rate: {args.rate} Hz")
    a(f"- channels: {args.channels}")
    a(f"- file: {args.output}")
    a("")
    a("## Recording result")
    a(f"- status: {'OK' if rec_ok else 'FAILED'}")
    a(f"- message: {rec_msg or 'none'}")
    if os.path.exists(args.output):
        a(f"- file size: {os.path.getsize(args.output)} bytes")
    a("")
    a("## Audio analysis")
    if stats.get("error"):
        a(f"- ERROR: {stats['error']}")
    else:
        a(f"- format: {stats['channels']}ch, "
          f"{stats['rate']}Hz, {stats['sampwidth']*8}-bit PCM")
        a(f"- duration: {stats['duration']:.2f}s ({stats['nframes']} frames)")
        a(f"- peak: {stats['peak']}/32768  ({stats['peak_dbfs']:.1f} dBFS)")
        a(f"- RMS:  {stats['rms']:.1f}/32768  ({stats['rms_dbfs']:.1f} dBFS)")
        a(f"- DC offset: {stats['dc_offset']:.2f}")
        a(f"- silence: {stats['silence_pct']:.1f}% of samples (below ~-40 dBFS)")
        a(f"- clipping: {stats['clip_pct']:.3f}% of samples")
        a(f"- est. SNR: {stats['snr_db']:.1f} dB")
        a(f"- VERDICT: {stats['verdict']}")
    a("")
    a("## Playback result")
    a(f"- method: {play_method}")
    a(f"- status: {'OK' if play_rc == 0 else 'FAILED/SKIPPED'}")
    a(f"- message: {play_msg}")
    a("")
    a("## Overall")
    mic_ok = bool(rec_ok and not stats.get("error") and stats.get("peak", 0) >= 300)
    play_ok = play_rc == 0
    a(f"- microphone usable: {'YES' if mic_ok else 'NO'}")
    a(f"- playback working:   {'YES' if play_ok else 'NO/UNKNOWN'}")
    a(f"- full mic->speaker loop: {'YES' if mic_ok and play_ok else 'NO'}")
    a(REPORT_MARKER_END)

    print()
    print("\n".join(lines))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="USB camera microphone checker")
    ap.add_argument("--duration", type=int, default=10,
                    help="seconds to record (default 10, must be >= 1)")
    ap.add_argument("--device", default=None,
                    help='ALSA capture device, e.g. "hw:0,0" (default: auto)')
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--channels", type=int, default=2, choices=[1, 2])
    ap.add_argument("--output", default="/tmp/usb_mic_check.wav")
    ap.add_argument("--no-playback", action="store_true")
    ap.add_argument("--pipewire-sink", default=None,
                    help="PipeWire sink node id for playback (default: auto)")
    args = ap.parse_args()

    if args.duration < 1:
        ap.error("--duration must be at least 1 second")
    if args.duration < 10 and "--duration" not in sys.argv:
        args.duration = 10  # enforce the >= 10s requirement unless explicitly overridden

    # Make sure PipeWire / WirePlumber clients (wpctl, pw-play) can find the
    # session bus + runtime dir even when launched from a minimal environment.
    os.environ.update(setup_pipewire_env())

    tools = [t for t in ("arecord", "aplay", "pw-play", "wpctl") if have(t)]
    log(f"[INFO] tools available: {', '.join(tools) if tools else 'none'}")

    if not have("arecord"):
        warn("arecord is not installed (alsa-utils). Cannot record.")
        print_report(args, None, None, "arecord missing", False,
                     "arecord not installed", {}, "none", 1,
                     "arecord missing", tools)
        return 1

    # detect device
    if args.device:
        device, card_num, dev_desc = args.device, "?", f"forced: {args.device}"
    else:
        device, card_num, dev_desc = detect_usb_mic()
    if not device:
        warn(dev_desc)
        print_report(args, None, None, dev_desc, False, dev_desc,
                     {}, "none", 1, "no device", tools)
        return 1
    log(f"[INFO] using capture device: {device}  ({dev_desc})")

    # record
    rec_ok, rec_msg = record(device, args.duration, args.rate, args.channels, args.output)

    # analyse
    stats = {}
    if rec_ok and os.path.exists(args.output):
        stats = analyse(args.output)
        if stats.get("error"):
            warn(stats["error"])
    else:
        stats = {"error": rec_msg or "no recording file"}

    # playback
    play_method, play_rc, play_msg = playback(
        args.output, args.pipewire_sink, args.no_playback
    )

    print_report(args, device, card_num, dev_desc, rec_ok, rec_msg,
                 stats, play_method, play_rc, play_msg, tools)

    mic_ok = bool(rec_ok and not stats.get("error") and stats.get("peak", 0) >= 300)
    return 0 if mic_ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INTERRUPTED]")
        sys.exit(130)
