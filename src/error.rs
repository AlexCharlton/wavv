use crate::chunk::ChunkTag;

/// Error type for different parsing failures
#[derive(Debug, PartialEq)]
pub enum Error {
    /// Unknown or unsupported Chunk ID
    UnknownChunkID([u8; 4]),
    /// Failed parsing slice into specific bytes
    CantParseSliceInto,
    /// Failed parsing chunk with given tag
    CantParseChunk(ChunkTag),
    /// No WAVE tag found
    NoWaveTagFound,
    /// No riff chunk found
    NoRiffChunkFound,
    /// No data chunk found
    NoDataChunkFound,
    /// No fmt/header chunk found
    NoFmtChunkFound,
    /// Unsupported bit depth
    UnsupportedBitDepth(u16),
    /// Unsupported format
    UnsupportedFormat(u16),
}

#[cfg(feature = "embedded")]
#[derive(Debug, PartialEq)]
pub enum ReadError<E> {
    /// Error from the underlying reader
    Reader(E),
    /// Error from the parser
    Parser(Error),
}
impl<E> From<Error> for ReadError<E> {
    fn from(e: Error) -> Self {
        ReadError::Parser(e)
    }
}
