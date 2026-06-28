use eyre::{eyre, Result, WrapErr};
use turbojpeg::{Compressor, Image, PixelFormat, Subsamp};

const RGB_CHANNELS: usize = 3;

pub(crate) struct JpegEncoder {
    compressor: Compressor,
}

impl JpegEncoder {
    pub(crate) fn new(quality: u8) -> Result<Self> {
        let mut compressor = Compressor::new().wrap_err("Failed to create TurboJPEG compressor")?;
        compressor
            .set_quality(i32::from(quality))
            .wrap_err("Failed to configure JPEG quality")?;
        compressor
            .set_subsamp(Subsamp::Sub2x1)
            .wrap_err("Failed to configure JPEG 4:2:2 subsampling")?;
        Ok(Self { compressor })
    }

    pub(crate) fn encode(&mut self, rgb: &[u8], width: u32, height: u32) -> Result<Vec<u8>> {
        let (width, pitch, height) = validate_rgb_frame(rgb, width, height)?;
        let image = Image {
            pixels: rgb,
            width,
            pitch,
            height,
            format: PixelFormat::RGB,
        };
        self.compressor
            .compress_to_vec(image)
            .wrap_err("JPEG encoding failed")
    }
}

fn validate_rgb_frame(rgb: &[u8], width: u32, height: u32) -> Result<(usize, usize, usize)> {
    if width == 0 || height == 0 {
        return Err(eyre!("Invalid frame dimensions: {width}x{height}"));
    }

    let width = usize::try_from(width).wrap_err("Frame width does not fit usize")?;
    let height = usize::try_from(height).wrap_err("Frame height does not fit usize")?;
    i32::try_from(width).wrap_err("Frame width exceeds TurboJPEG limit")?;
    i32::try_from(height).wrap_err("Frame height exceeds TurboJPEG limit")?;

    let pitch = width
        .checked_mul(RGB_CHANNELS)
        .ok_or_else(|| eyre!("RGB row size overflow for width {width}"))?;
    i32::try_from(pitch).wrap_err("RGB row size exceeds TurboJPEG limit")?;
    let expected = pitch
        .checked_mul(height)
        .ok_or_else(|| eyre!("RGB frame size overflow for {width}x{height}"))?;

    if rgb.len() != expected {
        return Err(eyre!(
            "Invalid RGB data size: expected {expected} bytes ({width}x{height}x3), got {} bytes",
            rgb.len()
        ));
    }

    Ok((width, pitch, height))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_zero_and_overflowing_dimensions() {
        assert!(validate_rgb_frame(&[], 0, 1).is_err());
        assert!(validate_rgb_frame(&[], 1, 0).is_err());
        assert!(validate_rgb_frame(&[], u32::MAX, u32::MAX).is_err());
    }

    #[test]
    fn rejects_incorrect_rgb_length() {
        let error = validate_rgb_frame(&[0; 11], 2, 2).unwrap_err();
        assert!(error.to_string().contains("expected 12 bytes"));
    }

    #[test]
    fn writes_jpeg_dimensions_and_422_subsampling() -> Result<()> {
        let mut encoder = JpegEncoder::new(80)?;
        let jpeg = encoder.encode(&test_rgb(16, 8, 0), 16, 8)?;

        assert!(jpeg.starts_with(&[0xff, 0xd8]));
        assert!(jpeg.ends_with(&[0xff, 0xd9]));
        let header = turbojpeg::read_header(&jpeg)?;
        assert_eq!((header.width, header.height), (16, 8));
        assert_eq!(header.subsamp, Subsamp::Sub2x1);
        Ok(())
    }

    #[test]
    fn reuses_compressor_across_frames() -> Result<()> {
        let mut encoder = JpegEncoder::new(80)?;
        let first = encoder.encode(&test_rgb(16, 8, 0), 16, 8)?;
        let second = encoder.encode(&test_rgb(16, 8, 31), 16, 8)?;

        assert_ne!(first, second);
        assert_eq!(turbojpeg::read_header(&second)?.subsamp, Subsamp::Sub2x1);
        Ok(())
    }

    fn test_rgb(width: usize, height: usize, offset: u8) -> Vec<u8> {
        (0..width * height * RGB_CHANNELS)
            .map(|index| (index as u8).wrapping_add(offset))
            .collect()
    }
}
