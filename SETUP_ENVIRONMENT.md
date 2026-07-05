# Setup Environment Guidelines

This document describes how to set up the system dependencies, permissions, and ALSA configurations to run the Robo-Fleet Dora nodes on the host or inside containers.

## 1. System Packages Installation

### Debian / Ubuntu
```shell
sudo apt update
sudo apt install -y build-essential cmake pkg-config libssl-dev
sudo apt install -y libglib2.0-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev gstreamer1.0-plugins-good
sudo apt install -y libespeak-ng-dev
sudo apt install -y libasound2-dev libasound2-plugins alsa-utils
```

### Fedora
```shell
sudo dnf install -y gcc gcc-c++ cmake pkg-config openssl-devel
sudo dnf install -y glib2-devel gstreamer1.0-devel gstreamer1.0-plugins-base-devel gstreamer1.0-plugins-good
sudo dnf install -y espeak-ng-devel
sudo dnf install -y alsa-lib-devel alsa-plugins-pulseaudio alsa-utils pipewire-alsa pipewire-pulseaudio
```

---

## 2. Permissions Setup

Processes capturing audio/video must have permission to access host devices (e.g., `/dev/snd/*` and `/dev/video*`).

```shell
# Add current user to video and audio groups
sudo usermod -aG video,audio ${USER}

# Restart user services manager so that group updates apply to PipeWire
sudo systemctl restart user@$(id -u)

# Activate the group in your current terminal session immediately
newgrp audio
newgrp video
```

---

## 3. PipeWire & PulseAudio Configuration on Host

For headless servers or SSH-based environments, you must keep PipeWire user services running and enable lingering so they don't terminate upon logout.

```shell
# Keep user services active after logout
loginctl enable-linger ${USER}

# Start and enable user-level audio daemons
export XDG_RUNTIME_DIR=/run/user/$(id -u)
systemctl --user enable --now pipewire pipewire-pulse wireplumber
```

---

## 4. ALSA Configuration (`~/.asoundrc`)

The `audio-capture` node requests a `16000` Hz sample rate and `f32` (Float32) sample format. Depending on your environment, choose the appropriate ALSA configuration for `~/.asoundrc` on the host:

### Option A: Workstation / Modern OS with PipeWire or PulseAudio
Raw PipeWire ALSA devices do not perform automatic format or sample rate conversion natively. You must wrap the default device in ALSA's `plug` converter plugin:
```shell
cat > ~/.asoundrc << 'EOF'
pcm.!default {
    type plug
    slave.pcm "pipewire"
}

ctl.!default {
    type pipewire
}
EOF
```

> [!IMPORTANT]
> **Headless / SSH / Non-Interactive Session Troubleshooting:**
> If you run Dora dataflows over SSH, via background scripts, or from system services, the `XDG_RUNTIME_DIR` environment variable might not be set.
> Since ALSA's PipeWire plugin relies on `XDG_RUNTIME_DIR` to locate the user's audio socket (e.g., `/run/user/1000/pipewire-0`), missing this variable results in the following warning/error:
> `ALSA function 'snd_pcm_open' failed with error 'Host is down (112)'`
>
> **Fixes & Workarounds:**
> 1. **Codebase Auto-recovery:** The codebase automatically detects missing `XDG_RUNTIME_DIR` at startup and maps it to `/run/user/<uid>` if that directory exists.
> 2. **Manual Setup:** If you are running tools manually outside of standard entrypoints, export it in your terminal session before starting Dora:
>    ```shell
>    export XDG_RUNTIME_DIR=/run/user/$(id -u)
>    ```


### Option B: Headless Raspberry Pi 4 / Raw ALSA (No Sound Server)
If you are running on a Raspberry Pi 4 host without a running audio server (PipeWire/PulseAudio), route the ALSA default device directly to your USB microphone hardware (usually card 2 on Pi):
```shell
cat > ~/.asoundrc << 'EOF'
pcm.!default {
    type hw
    card 2
}

ctl.!default {
    type hw
    card 2
}
EOF
```
*(Tip: If your USB microphone driver does not support `f32` format natively, wrap it in a plug plugin by changing `type hw` to `type plug` and `card 2` to `slave.pcm "hw:2,0"`).*

---

## 5. Running inside Docker / Podman (Current Workstation Path)

Phase 10 validated the local container workflow on Fedora x86_64 with Podman's
Docker-compatible CLI, host Pulse socket mounting, and the workstation compose
override.

Use the verified startup path:
```shell
export XDG_RUNTIME_DIR=/run/user/$(id -u)
docker compose \
  -f docker/docker-compose.yml \
  -f docker/docker-compose.workstation.yml \
  --profile mongodb --profile orchestra --profile rover-kiwi \
  up -d --build
```

Notes:
- The workstation override uses `group_add: keep-groups`, so a manual `AUDIO_GID` export is not part of the verified path.
- The current host required `WORKSTATION_AUDIO_DEVICE=sysdefault:CARD=Camera` for stable rover capture in the container.
- If you launch containers outside the provided entrypoints, keep `XDG_RUNTIME_DIR` aligned with `/run/user/<uid>` so Pulse/ALSA discovery can find the mounted runtime socket.

---

## 6. Runtime Notes

* `SOURCE_FPS` controls capture cadence.
* `VIEW_STREAM_FPS` controls rover-side JPEG publish cadence.
* Keep `SOURCE_FPS` aligned with the camera and set `VIEW_STREAM_FPS` for the desired viewer rate.
* Central STT uses startup-only `STT_PROFILE` (`en-vad-offline` or `vi-vad-offline`) plus
  `STT_MODEL_ROOT` pointing at the Sherpa ASR bundle root.
* Rover voice output is the `edge_voice` Supertonic path. Playback routing and
  source-aware mic suppression are active in the current dataflows.
