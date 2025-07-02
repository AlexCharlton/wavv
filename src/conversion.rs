/// Trait for converting WAV samples to a generic numeric type
pub trait FromWavSample {
    /// Convert a u8 sample to the target type
    fn from_u8(sample: u8) -> Self;
    /// Convert a i16 sample to the target type
    fn from_i16(sample: i16) -> Self;
    /// Convert a i32 sample to the target type
    fn from_i32(sample: i32) -> Self;
}

// Implement for common numeric types
impl FromWavSample for f32 {
    fn from_u8(sample: u8) -> Self {
        (sample as f32 - 128.0) / 128.0
    }
    fn from_i16(sample: i16) -> Self {
        sample as f32 / 32768.0
    }
    fn from_i32(sample: i32) -> Self {
        sample as f32 / 8388608.0
    }
}

impl FromWavSample for f64 {
    fn from_u8(sample: u8) -> Self {
        (sample as f64 - 128.0) / 128.0
    }
    fn from_i16(sample: i16) -> Self {
        sample as f64 / 32768.0
    }
    fn from_i32(sample: i32) -> Self {
        sample as f64 / 8388608.0
    }
}

impl FromWavSample for u8 {
    fn from_u8(sample: u8) -> Self {
        sample
    }
    fn from_i16(sample: i16) -> Self {
        ((sample as f32 / 32768.0) * 255.0 + 128.0).clamp(0.0, 255.0) as u8
    }
    fn from_i32(sample: i32) -> Self {
        ((sample as f32 / 8388608.0) * 255.0 + 128.0).clamp(0.0, 255.0) as u8
    }
}

impl FromWavSample for i16 {
    fn from_u8(sample: u8) -> Self {
        ((sample as f32 - 128.0) / 128.0 * 32768.0).clamp(-32768.0, 32767.0) as i16
    }
    fn from_i16(sample: i16) -> Self {
        sample
    }
    fn from_i32(sample: i32) -> Self {
        (sample as f32 / 8388608.0 * 32768.0).clamp(-32768.0, 32767.0) as i16
    }
}

impl FromWavSample for i32 {
    fn from_u8(sample: u8) -> Self {
        ((sample as f32 - 128.0) / 128.0 * 8388608.0).clamp(-8388608.0, 8388607.0) as i32
    }
    fn from_i16(sample: i16) -> Self {
        sample as i32
    }
    fn from_i32(sample: i32) -> Self {
        sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_conversions() {
        assert_eq!(f32::from_u8(128), 0.0);
        assert_eq!(f32::from_u8(255), 0.9921875);
        assert_eq!(f32::from_u8(0), -1.0);

        assert_eq!(f32::from_i16(0), 0.0);
        assert_eq!(f32::from_i16(32767), 0.9999695);
        assert_eq!(f32::from_i16(-32768), -1.0);

        assert_eq!(f32::from_i32(0), 0.0);
        assert_eq!(f32::from_i32(8388607), 0.9999999);
        assert_eq!(f32::from_i32(-8388608), -1.0);
    }

    #[test]
    fn test_i32_conversions() {
        assert_eq!(i32::from_u8(128), 0);
        assert_eq!(i32::from_i16(0), 0);
        assert_eq!(i32::from_i32(12345), 12345);
    }
}
