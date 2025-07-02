use crate::chunk::{Chunk, ChunkTag};
use crate::error::Error;
use alloc::vec;
use core::convert::TryInto;

/// Audio format types supported by the WAV format
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum AudioFormat {
    /// PCM (Pulse Code Modulation) - integer samples
    Pcm = 1,
    /// IEEE float - floating point samples
    IeeeFloat = 3,
}

impl AudioFormat {
    fn from_u16(value: u16) -> Result<Self, Error> {
        match value {
            1 => Ok(AudioFormat::Pcm),
            3 => Ok(AudioFormat::IeeeFloat),
            _ => Err(Error::UnsupportedFormat(value)),
        }
    }

    pub(crate) fn to_u16(self) -> u16 {
        self as u16
    }
}

/// Struct representing the `fmt_` section of a WAV file
///
/// for more information see [`here`]
///
/// [`here`]: http://soundfile.sapp.org/doc/WaveFormat/
pub struct Fmt {
    /// audio format, PCM or IEEE float
    pub audio_format: AudioFormat,
    /// sample rate, typical values are `44_100`, `48_000` or `96_000`
    pub sample_rate: u32,
    /// number of audio channels in the sample data, channels are interleaved
    pub num_channels: u16,
    /// bit depth for each sample, typical values are `16`, `24`, or `32`
    pub bit_depth: u16,
}

impl Fmt {
    pub(crate) fn from_chunk(chunk: &Chunk) -> Result<Self, Error> {
        let audio_format = AudioFormat::from_u16(
            chunk.bytes[0..2]
                .try_into()
                .map_err(|_| Error::CantParseSliceInto)
                .map(|b| u16::from_le_bytes(b))?,
        )?;

        // Validate format matches bit depth
        if audio_format == AudioFormat::IeeeFloat
            && chunk.bytes[14..16]
                .try_into()
                .map_err(|_| Error::CantParseSliceInto)
                .map(|b| u16::from_le_bytes(b))?
                != 32
        {
            return Err(Error::UnsupportedFormat(audio_format.to_u16()));
        }

        let num_channels = chunk.bytes[2..4]
            .try_into()
            .map_err(|_| Error::CantParseSliceInto)
            .map(|b| u16::from_le_bytes(b))?;

        let sample_rate = chunk.bytes[4..8]
            .try_into()
            .map_err(|_| Error::CantParseSliceInto)
            .map(|b| u32::from_le_bytes(b))?;

        let bit_depth = chunk.bytes[14..16]
            .try_into()
            .map_err(|_| Error::CantParseSliceInto)
            .map(|b| u16::from_le_bytes(b))?;

        Ok(Fmt {
            audio_format,
            num_channels,
            sample_rate,
            bit_depth,
        })
    }

    pub(crate) fn to_chunk(&self) -> Chunk {
        let br = ((self.sample_rate * (self.bit_depth as u32) * (self.num_channels as u32)) / 8)
            .to_le_bytes();
        let ba = ((self.num_channels * self.bit_depth) / 8).to_le_bytes();
        let nc = self.num_channels.to_le_bytes();
        let sr = self.sample_rate.to_le_bytes();
        let bd = self.bit_depth.to_le_bytes();
        let af = self.audio_format.to_u16().to_le_bytes();

        let bytes = vec![
            af[0], af[1], // audio format
            nc[0], nc[1], // num channels
            sr[0], sr[1], sr[2], sr[3], // sample rate
            br[0], br[1], br[2], br[3], // byte rate
            ba[0], ba[1], // block align
            bd[0], bd[1], // bits per sample
        ];

        Chunk {
            id: ChunkTag::Fmt,
            bytes,
        }
    }
}
