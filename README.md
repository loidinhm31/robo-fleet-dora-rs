OFF_SCREEN() {
    sudo sh -c '
    dev=/sys/class/backlight/amdgpu_bl1
    level=$(cat "$dev/brightness") || exit 1
    
          # Preserve the last non-zero brightness.
          [ "$level" -eq 0 ] ||
            printf "%s\n" "$level" > /run/screen-brightness
    
          TERM=linux setterm \
            --blank=force \
            --powersave=powerdown \
            < /dev/tty1 > /dev/tty1
    
          printf "0\n" > "$dev/brightness"
          printf "4\n" > "$dev/bl_power"
    '
}

ON_SCREEN() {
    sudo sh -c '
    dev=/sys/class/backlight/amdgpu_bl1
    max=$(cat "$dev/max_brightness") || exit 1
    level=$(cat /run/screen-brightness 2>/dev/null ||
    printf "%s\n" "$max")
    
          case "$level" in
            ""|*[!0-9]*) level=$max ;;
          esac
    
          [ "$level" -le "$max" ] || level=$max
          [ "$level" -gt 0 ] || level=$max
    
          printf "0\n" > "$dev/bl_power"
          printf "%s\n" "$level" > "$dev/brightness"
    
          TERM=linux setterm \
            --powersave=off \
            --blank=poke \
            < /dev/tty1 > /dev/tty1
        '

}