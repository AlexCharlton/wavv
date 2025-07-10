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

#[cfg(feature = "io")]
#[derive(Debug, PartialEq)]
pub enum ReadError<E: core::fmt::Debug> {
    /// Error from the underlying reader
    Reader(E),
    /// Error from the parser
    Parser(Error),
    /// Unexpected EOF
    UnexpectedEof,
}

#[cfg(feature = "std")]
impl<E: core::fmt::Debug> std::error::Error for ReadError<E> {}

#[cfg(feature = "std")]
impl<E: core::fmt::Debug> std::fmt::Display for ReadError<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

#[cfg(feature = "io")]
impl<E: core::fmt::Debug> From<Error> for ReadError<E> {
    fn from(e: Error) -> Self {
        ReadError::Parser(e)
    }
}

#[cfg(feature = "io")]
impl<E: core::fmt::Debug> From<embedded_io_async::ReadExactError<E>> for ReadError<E> {
    fn from(e: embedded_io_async::ReadExactError<E>) -> Self {
        match e {
            embedded_io_async::ReadExactError::UnexpectedEof => ReadError::UnexpectedEof,
            embedded_io_async::ReadExactError::Other(e) => ReadError::Reader(e),
        }
    }
}
