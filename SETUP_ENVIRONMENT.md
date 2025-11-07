```shell
sudo apt install build-essential cmake pkg-config

sudo apt install libglib2.0-dev pkg-config libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev

sudo apt install gstreamer1.0-plugins-good

sudo apt install libssl-dev

sudo apt install libespeak-ng-dev
```


Create ALSA config for microphone USB
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