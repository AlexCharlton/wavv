/// Trait for converting WAV samples to a generic numeric type
pub trait FromWavSample: Sized + Copy {
    /// Convert a u8 sample to the target type
    fn from_u8(sample: u8) -> Self;
    /// Convert a i16 sample to the target type
    fn from_i16(sample: i16) -> Self;
    /// Convert a i32 sample to the target type
    fn from_i32(sample: i32) -> Self;
    /// Convert a f32 sample to the target type
    fn from_f32(sample: f32) -> Self;

    /// The zero value for the target type
    fn zero() -> Self;
    /// Add two samples together without overflow
    fn add(self, other: Self) -> Self;
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
        sample as f32 / 2_147_483_648.0
    }
    fn from_f32(sample: f32) -> Self {
        sample
    }

    fn zero() -> Self {
        0.0
    }
    fn add(self, other: Self) -> Self {
        self + other
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
        sample as f64 / 2_147_483_648.0
    }
    fn from_f32(sample: f32) -> Self {
        sample as f64
    }

    fn zero() -> Self {
        0.0
    }
    fn add(self, other: Self) -> Self {
        self + other
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
        ((sample as f32 / 2_147_483_648.0) * 255.0 + 128.0).clamp(0.0, 255.0) as u8
    }
    fn from_f32(sample: f32) -> Self {
        ((sample + 1.0) * 127.5).clamp(0.0, 255.0) as u8
    }

    fn zero() -> Self {
        0
    }
    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
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
        (sample >> 16) as i16
    }
    fn from_f32(sample: f32) -> Self {
        (sample * 32767.5).clamp(-32767.5, 32767.5) as i16
    }

    fn zero() -> Self {
        0
    }
    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
    }
}

impl FromWavSample for i32 {
    fn from_u8(sample: u8) -> Self {
        (((sample as f32 - 128.0) / 128.0 * 2_147_483_648.0)
            .clamp(-2_147_483_648.0, 2_147_483_647.0) as i32)
            << 8
    }
    fn from_i16(sample: i16) -> Self {
        (sample as i32) << 16
    }
    fn from_i32(sample: i32) -> Self {
        sample
    }
    fn from_f32(sample: f32) -> Self {
        (sample * 2_147_483_647.5).clamp(-2_147_483_647.5, 2_147_483_647.5) as i32
    }

    fn zero() -> Self {
        0
    }
    fn add(self, other: Self) -> Self {
        self.saturating_add(other)
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
        assert_eq!(f32::from_i32(2147483647), 1.0);
        assert_eq!(f32::from_i32(-2147483648), -1.0);
    }

    #[test]
    fn test_i32_conversions() {
        assert_eq!(i32::from_u8(128), 0);
        assert_eq!(i32::from_i16(0), 0);
        assert_eq!(i32::from_i32(12345), 12345);
    }

    #[test]
    fn test_f32_from_f32_conversions() {
        assert_eq!(f32::from_f32(0.0), 0.0);
        assert_eq!(f32::from_f32(0.5), 0.5);
        assert_eq!(f32::from_f32(-0.5), -0.5);
        assert_eq!(f32::from_f32(1.0), 1.0);
        assert_eq!(f32::from_f32(-1.0), -1.0);
    }

    #[test]
    fn test_f64_conversions() {
        assert_eq!(f64::from_f32(0.0), 0.0);
        assert_eq!(f64::from_f32(0.5), 0.5);
        assert_eq!(f64::from_f32(-0.5), -0.5);
        assert_eq!(f64::from_f32(1.0), 1.0);
        assert_eq!(f64::from_f32(-1.0), -1.0);
    }

    #[test]
    fn test_u8_conversions() {
        assert_eq!(u8::from_f32(0.0), 127);
        assert_eq!(u8::from_f32(1.0), 255);
        assert_eq!(u8::from_f32(-1.0), 0);
    }

    #[test]
    fn test_i16_conversions() {
        assert_eq!(i16::from_f32(0.0), 0);
        assert_eq!(i16::from_f32(1.0), 32767);
        assert_eq!(i16::from_f32(-1.0), -32767);
    }

    #[test]
    fn test_i32_conversions_from_f32() {
        assert_eq!(i32::from_f32(0.0), 0);
        assert_eq!(i32::from_f32(1.0), 2_147_483_647);
        assert_eq!(i32::from_f32(-1.0), -2_147_483_648);
    }
}
