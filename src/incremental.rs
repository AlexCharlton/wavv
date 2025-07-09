use crate::chunk::{Chunk, ChunkTag};
use crate::conversion::FromWavSample;
use crate::error::Error;
use crate::fmt::{AudioFormat, Fmt};
use alloc::vec;
use alloc::vec::Vec;
use core::convert::TryInto;

use crate::error::ReadError;

/// Iterator for reading WAV data incrementally from a reader
pub struct IncrementalWavIterator<T, R> {
    reader: R,
    fmt: Fmt,
    bytes_per_sample: usize,
    buffer: Vec<u8>,
    buffer_size: usize,
    read_pos: usize,       // Position where we read new data from reader
    write_pos: usize,      // Position where we write new data to buffer
    data_available: usize, // Amount of data available in buffer
    _phantom: core::marker::PhantomData<T>,
}

impl<T, R> IncrementalWavIterator<T, R>
where
    R: embedded_io::Read,
    T: FromWavSample + Copy,
{
    fn new(reader: R, fmt: Fmt, buffer_size: usize, audio_data: Vec<u8>) -> Self {
        let bytes_per_sample = (fmt.bit_depth / 8) as usize;
        let mut buffer = vec![0; buffer_size];

        // Copy initial audio data into the circular buffer
        let initial_data_len = audio_data.len().min(buffer_size);
        buffer[..initial_data_len].copy_from_slice(&audio_data[..initial_data_len]);

        Self {
            reader,
            fmt,
            bytes_per_sample,
            buffer,
            buffer_size,
            read_pos: 0,
            write_pos: initial_data_len % buffer_size,
            data_available: initial_data_len,
            _phantom: core::marker::PhantomData,
        }
    }

    fn read_more_data(&mut self) -> Result<bool, ReadError<R::Error>> {
        // Calculate how much space is available in the buffer
        let space_available = self.buffer_size - self.data_available;

        if space_available == 0 {
            // Buffer is full, we need to wait for more data to be consumed
            return Ok(false);
        }

        // Calculate how much we can read
        let read_amount = if self.write_pos >= self.read_pos {
            // Write position is ahead of read position
            let space_to_end = self.buffer_size - self.write_pos;
            space_to_end.min(space_available)
        } else {
            // Write position is behind read position
            (self.read_pos - self.write_pos).min(space_available)
        };

        if read_amount == 0 {
            return Ok(false);
        }

        // Read data into the buffer
        let slice = &mut self.buffer[self.write_pos..self.write_pos + read_amount];

        match self.reader.read(slice) {
            Ok(0) => Ok(false),
            Ok(n) => {
                self.write_pos = (self.write_pos + n) % self.buffer_size;
                self.data_available += n;

                Ok(true)
            }
            Err(e) => Err(ReadError::Reader(e)),
        }
    }

    fn read_sample(&mut self) -> Result<Option<T>, ReadError<R::Error>> {
        // Ensure we have enough data for a complete sample
        while self.data_available < self.bytes_per_sample {
            if !self.read_more_data()? {
                return Ok(None); // EOF
            }
        }

        let sample = match self.fmt.bit_depth {
            8 => {
                let sample = self.buffer[self.read_pos];
                self.read_pos = (self.read_pos + 1) % self.buffer_size;
                self.data_available -= 1;
                T::from_u8(sample)
            }
            16 => {
                let sample = if self.read_pos + 1 < self.buffer_size {
                    // No wrap-around needed
                    let sample = i16::from_le_bytes([
                        self.buffer[self.read_pos],
                        self.buffer[self.read_pos + 1],
                    ]);
                    self.read_pos = (self.read_pos + 2) % self.buffer_size;
                    self.data_available -= 2;
                    sample
                } else {
                    // Wrap-around needed
                    let sample = i16::from_le_bytes([self.buffer[self.read_pos], self.buffer[0]]);
                    self.read_pos = 1;
                    self.data_available -= 2;
                    sample
                };
                T::from_i16(sample)
            }
            24 => {
                let sample = if self.read_pos + 2 < self.buffer_size {
                    // No wrap-around needed
                    let sign = self.buffer[self.read_pos + 2] >> 7;
                    let sign_byte = if sign == 1 { 0xff } else { 0x0 };
                    let sample = i32::from_le_bytes([
                        self.buffer[self.read_pos],
                        self.buffer[self.read_pos + 1],
                        self.buffer[self.read_pos + 2],
                        sign_byte,
                    ]);
                    self.read_pos = (self.read_pos + 3) % self.buffer_size;
                    self.data_available -= 3;
                    sample
                } else {
                    // Wrap-around needed
                    let bytes = [
                        self.buffer[self.read_pos],
                        self.buffer[(self.read_pos + 1) % self.buffer_size],
                        self.buffer[(self.read_pos + 2) % self.buffer_size],
                    ];
                    let sign = bytes[2] >> 7;
                    let sign_byte = if sign == 1 { 0xff } else { 0x0 };
                    let sample = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], sign_byte]);
                    self.read_pos = (self.read_pos + 3) % self.buffer_size;
                    self.data_available -= 3;
                    sample
                };
                T::from_i32(sample)
            }
            32 => {
                let sample = if self.read_pos + 3 < self.buffer_size {
                    // No wrap-around needed
                    let sample = f32::from_le_bytes([
                        self.buffer[self.read_pos],
                        self.buffer[self.read_pos + 1],
                        self.buffer[self.read_pos + 2],
                        self.buffer[self.read_pos + 3],
                    ]);
                    self.read_pos = (self.read_pos + 4) % self.buffer_size;
                    self.data_available -= 4;
                    sample
                } else {
                    // Wrap-around needed
                    let bytes = [
                        self.buffer[self.read_pos],
                        self.buffer[(self.read_pos + 1) % self.buffer_size],
                        self.buffer[(self.read_pos + 2) % self.buffer_size],
                        self.buffer[(self.read_pos + 3) % self.buffer_size],
                    ];
                    let sample = f32::from_le_bytes(bytes);
                    self.read_pos = (self.read_pos + 4) % self.buffer_size;
                    self.data_available -= 4;
                    sample
                };
                T::from_f32(sample)
            }
            _ => {
                return Err(ReadError::Parser(Error::UnsupportedBitDepth(
                    self.fmt.bit_depth,
                )))
            }
        };

        Ok(Some(sample))
    }
}

impl<T, R> Iterator for IncrementalWavIterator<T, R>
where
    R: embedded_io::Read,
    T: FromWavSample + Copy,
{
    type Item = Result<T, ReadError<R::Error>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.read_sample() {
            Ok(Some(sample)) => Some(Ok(sample)),
            Ok(None) => None, // EOF
            Err(e) => Some(Err(e)),
        }
    }
}

/// Incremental WAV struct that only contains format information
/// and allows incremental reading of audio data
pub struct IncrementalWav<R> {
    /// Contains data from the fmt chunk / header part of the file
    pub fmt: Fmt,
    /// The reader for reading additional audio data (if any)
    reader: R,
    /// Audio data that was already read during format parsing
    audio_data: Vec<u8>,
    /// Buffer size for incremental reading
    buffer_size: usize,
}

impl<R> IncrementalWav<R>
where
    R: embedded_io::Read,
{
    /// Create a incremental WAV from a reader, only reading the format information
    pub fn from_reader(mut reader: R, buffer_size: usize) -> Result<Self, ReadError<R::Error>> {
        // Read data into a buffer for chunk parsing
        let mut parse_buffer = vec![0; buffer_size];
        let mut total_read = 0;

        // Read initial data to find RIFF header
        let mut initial_read = 0;
        while initial_read < 12 {
            match reader.read(&mut parse_buffer[initial_read..]) {
                Ok(0) => return Err(ReadError::Parser(Error::NoRiffChunkFound)),
                Ok(n) => {
                    initial_read += n;
                    total_read += n;
                }
                Err(e) => return Err(ReadError::Reader(e)),
            }
        }
        let mut buffer_pos = 12;

        // Verify RIFF header
        if &parse_buffer[0..4] != b"RIFF" {
            return Err(ReadError::Parser(Error::NoRiffChunkFound));
        }

        // Verify WAVE identifier
        if &parse_buffer[8..12] != b"WAVE" {
            return Err(ReadError::Parser(Error::NoWaveTagFound));
        }

        let mut found_fmt = false;
        let mut found_data = false;
        let mut fmt = None;
        let mut audio_data = vec![];

        // Parse chunks from the buffer
        while !found_fmt || !found_data {
            // Ensure we have enough data for chunk header
            if buffer_pos + 8 > total_read {
                let needed = buffer_pos + 8 - total_read;
                let available = buffer_size - total_read;
                let to_read = needed.min(available);

                if to_read == 0 {
                    // Buffer is full, we need to shift data
                    let shift_amount = buffer_pos;
                    parse_buffer.copy_within(shift_amount..total_read, 0);
                    total_read -= shift_amount;
                    buffer_pos = 0;

                    // Read more data
                    let to_read = (buffer_size - total_read).min(needed);
                    match reader.read(&mut parse_buffer[total_read..total_read + to_read]) {
                        Ok(0) => break, // EOF
                        Ok(n) => total_read += n,
                        Err(e) => return Err(ReadError::Reader(e)),
                    }
                } else {
                    match reader.read(&mut parse_buffer[total_read..total_read + to_read]) {
                        Ok(0) => break, // EOF
                        Ok(n) => total_read += n,
                        Err(e) => return Err(ReadError::Reader(e)),
                    }
                }
            }

            // Parse chunk header
            let chunk_tag =
                ChunkTag::from_bytes(&parse_buffer[buffer_pos..buffer_pos + 4].try_into().unwrap());
            let chunk_size = u32::from_le_bytes(
                parse_buffer[buffer_pos + 4..buffer_pos + 8]
                    .try_into()
                    .unwrap(),
            ) as usize;

            match chunk_tag {
                ChunkTag::Fmt => {
                    // Ensure we have enough data for fmt chunk
                    if buffer_pos + 8 + chunk_size > total_read {
                        let needed = buffer_pos + 8 + chunk_size - total_read;
                        let available = buffer_size - total_read;
                        let to_read = needed.min(available);

                        if to_read == 0 {
                            // Buffer is full, we need to shift data
                            let shift_amount = buffer_pos;
                            parse_buffer.copy_within(shift_amount..total_read, 0);
                            total_read -= shift_amount;
                            buffer_pos = 0;

                            // Read more data
                            let to_read = (buffer_size - total_read).min(needed);
                            match reader.read(&mut parse_buffer[total_read..total_read + to_read]) {
                                Ok(0) => return Err(ReadError::Parser(Error::NoFmtChunkFound)),
                                Ok(n) => total_read += n,
                                Err(e) => return Err(ReadError::Reader(e)),
                            }
                        } else {
                            match reader.read(&mut parse_buffer[total_read..total_read + to_read]) {
                                Ok(0) => return Err(ReadError::Parser(Error::NoFmtChunkFound)),
                                Ok(n) => total_read += n,
                                Err(e) => return Err(ReadError::Reader(e)),
                            }
                        }
                    }

                    // Parse fmt chunk
                    let chunk_data =
                        parse_buffer[buffer_pos + 8..buffer_pos + 8 + chunk_size].to_vec();
                    let chunk = Chunk {
                        id: chunk_tag,
                        bytes: chunk_data,
                    };
                    let parsed_fmt = Fmt::from_chunk(&chunk)?;

                    // Validate audio format compatibility
                    match parsed_fmt.audio_format {
                        AudioFormat::Pcm => {
                            if parsed_fmt.bit_depth != 8
                                && parsed_fmt.bit_depth != 16
                                && parsed_fmt.bit_depth != 24
                            {
                                return Err(ReadError::Parser(Error::UnsupportedBitDepth(
                                    parsed_fmt.bit_depth,
                                )));
                            }
                        }
                        AudioFormat::IeeeFloat => {
                            if parsed_fmt.bit_depth != 32 {
                                return Err(ReadError::Parser(Error::UnsupportedFormat(
                                    parsed_fmt.audio_format.to_u16(),
                                )));
                            }
                        }
                    }

                    fmt = Some(parsed_fmt);
                    found_fmt = true;

                    // Move past this chunk
                    buffer_pos += 8 + chunk_size;

                    // Handle padding
                    if chunk_size % 2 == 1 {
                        buffer_pos += 1;
                    }
                }
                ChunkTag::Data => {
                    // Found data chunk - store remaining buffer data as audio_data
                    found_data = true;
                    audio_data = parse_buffer[buffer_pos + 8..total_read].to_vec();
                    break;
                }
                _ => {
                    // Skip other chunks
                    buffer_pos += 8 + chunk_size;

                    // Handle padding
                    if chunk_size % 2 == 1 {
                        buffer_pos += 1;
                    }
                }
            }
        }

        if !found_fmt {
            return Err(ReadError::Parser(Error::NoFmtChunkFound));
        }
        if !found_data {
            return Err(ReadError::Parser(Error::NoDataChunkFound));
        }

        Ok(IncrementalWav {
            fmt: fmt.unwrap(),
            reader,
            audio_data,
            buffer_size,
        })
    }

    /// Create a incremental WAV from a reader with a default buffer size
    pub fn from_reader_default(reader: R) -> Result<Self, ReadError<R::Error>> {
        Self::from_reader(reader, 4096) // Default 4KB buffer
    }

    /// Create an iterator for reading audio data incrementally
    pub fn iter_data<T>(self) -> IncrementalWavIterator<T, R>
    where
        T: FromWavSample + Copy,
    {
        IncrementalWavIterator::new(self.reader, self.fmt, self.buffer_size, self.audio_data)
    }
}

#[cfg(feature = "std")]
impl IncrementalWav<File> {
    /// Create a incremental WAV from a file path.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, ReadError<FileError>> {
        let file = std::fs::File::open(path).map_err(|e| ReadError::Reader(FileError(e)))?;
        Self::from_reader(File(file), 4096)
    }
}

#[cfg(feature = "std")]
mod file_wrapper {
    use std::fs;
    use std::io::Read;

    /// Wrapper for std::fs::File. Will be part of the  type returned by [IncrementalWav::from_file]
    pub struct File(pub fs::File);

    #[doc(hidden)]
    #[derive(Debug)]
    pub struct FileError(pub std::io::Error);

    impl embedded_io::Error for FileError {
        fn kind(&self) -> embedded_io::ErrorKind {
            embedded_io::ErrorKind::Other
        }
    }

    impl embedded_io::ErrorType for File {
        type Error = FileError;
    }

    impl embedded_io::Read for File {
        fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
            self.0.read(buf).map_err(|e| FileError(e))
        }
    }
}

#[cfg(feature = "std")]
pub use file_wrapper::{File, FileError};

//-----------------------------------
// MARK: Async

/// Async version of IncrementalWav
pub mod asynch {
    use super::*;
    use embedded_io_async::SeekFrom;

    /// Async version of IncrementalWav
    pub struct IncrementalWav<R> {
        /// Contains data from the fmt chunk / header part of the file
        pub fmt: Fmt,
        /// The reader for reading additional audio data (if any)
        reader: R,
        data_pos: usize,
        data_len: usize,
    }

    impl<R> IncrementalWav<R>
    where
        R: embedded_io_async::Read + embedded_io_async::Seek,
    {
        /// Create a incremental WAV from a reader, only reading the format information
        pub async fn from_reader(mut reader: R) -> Result<Self, ReadError<R::Error>> {
            let mut read_buffer = vec![0; 24];

            // Read initial data to find RIFF header
            match reader.read_exact(&mut read_buffer[..12]).await {
                Ok(_) => {}
                Err(embedded_io_async::ReadExactError::UnexpectedEof) => {
                    return Err(ReadError::Parser(Error::NoRiffChunkFound));
                }
                Err(embedded_io_async::ReadExactError::Other(e)) => {
                    return Err(ReadError::Reader(e))
                }
            }

            // Verify RIFF header
            if &read_buffer[0..4] != b"RIFF" {
                return Err(ReadError::Parser(Error::NoRiffChunkFound));
            }

            // Verify WAVE identifier
            if &read_buffer[8..12] != b"WAVE" {
                return Err(ReadError::Parser(Error::NoWaveTagFound));
            }

            let mut fmt = None;
            let data_pos;
            let data_len;

            // Find fmt and data chunks
            loop {
                match reader.read_exact(&mut read_buffer[..8]).await {
                    Ok(_) => {}
                    Err(embedded_io_async::ReadExactError::UnexpectedEof) => {
                        return Err(ReadError::Parser(Error::NoFmtChunkFound));
                    }
                    Err(embedded_io_async::ReadExactError::Other(e)) => {
                        return Err(ReadError::Reader(e))
                    }
                };

                // Parse chunk header
                let chunk_tag = ChunkTag::from_bytes(&read_buffer[0..4].try_into().unwrap());
                let chunk_size = u32::from_le_bytes(read_buffer[4..8].try_into().unwrap()) as usize;

                match chunk_tag {
                    ChunkTag::Fmt => {
                        // Read fmt chunk
                        reader.read_exact(&mut read_buffer[8..24]).await?;

                        // Parse fmt chunk
                        let chunk_data = read_buffer[8..24].to_vec();
                        let chunk = Chunk {
                            id: chunk_tag,
                            bytes: chunk_data,
                        };
                        let parsed_fmt = Fmt::from_chunk(&chunk)?;

                        // Validate audio format compatibility
                        match parsed_fmt.audio_format {
                            AudioFormat::Pcm => {
                                if parsed_fmt.bit_depth != 8
                                    && parsed_fmt.bit_depth != 16
                                    && parsed_fmt.bit_depth != 24
                                {
                                    return Err(ReadError::Parser(Error::UnsupportedBitDepth(
                                        parsed_fmt.bit_depth,
                                    )));
                                }
                            }
                            AudioFormat::IeeeFloat => {
                                if parsed_fmt.bit_depth != 32 {
                                    return Err(ReadError::Parser(Error::UnsupportedFormat(
                                        parsed_fmt.audio_format.to_u16(),
                                    )));
                                }
                            }
                        }

                        fmt = Some(parsed_fmt);

                        // Move past this chunk
                        if chunk_size > 16 {
                            reader
                                .seek(SeekFrom::Current(chunk_size as i64 - 16))
                                .await
                                .map_err(|e| ReadError::Reader(e))?;
                        }
                        continue;
                    }
                    ChunkTag::Data => {
                        data_pos = reader
                            .stream_position()
                            .await
                            .map_err(|e| ReadError::Reader(e))?;
                        data_len = chunk_size as u64;
                        break;
                    }
                    _ => {
                        // Skip other chunks
                        reader
                            .seek(SeekFrom::Current(chunk_size as i64))
                            .await
                            .map_err(|e| ReadError::Reader(e))?;
                    }
                }
            }

            if fmt.is_none() {
                return Err(ReadError::Parser(Error::NoFmtChunkFound));
            }
            if data_len == 0 {
                return Err(ReadError::Parser(Error::NoDataChunkFound));
            }

            Ok(IncrementalWav {
                fmt: fmt.unwrap(),
                reader,
                data_pos: data_pos as usize,
                data_len: data_len as usize,
            })
        }

        /// Get the number of samples * num_channels in the WAV file. In other words, this is the number of numerical values in the WAV file.
        pub fn num_sample_values(&self) -> usize {
            let bytes_per_sample = self.fmt.bit_depth as usize / 8;
            self.data_len as usize / bytes_per_sample
        }

        /// Read 8-bit values from the WAV file from the start position (in number of samples * num_channels) to the number of values in the buffer. The start position plus the length of the buffer must be less than the total number of (samples * num_channels) in the WAV file.
        pub async fn read_u8(
            &mut self,
            from: usize,
            buf: &mut [u8],
        ) -> Result<(), ReadError<R::Error>> {
            if self.fmt.bit_depth == 8 {
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(buf).await?;
                Ok(())
            } else if self.fmt.bit_depth == 16 {
                let mut samples = vec![0; buf.len() * 2];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from * 2) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    let i16_i = i * 2;
                    let i16_sample = i16::from_le_bytes([samples[i16_i], samples[i16_i + 1]]);
                    let u8_sample = u8::from_i16(i16_sample);
                    buf[i] = u8_sample;
                    i += 1;
                }
                Ok(())
            } else if self.fmt.bit_depth == 24 {
                let mut samples = vec![0; buf.len() * 3];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from * 3) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    let i24_i = i * 3;
                    let sign = samples[i24_i + 2] >> 7;
                    let sign_byte = if sign == 1 { 0xff } else { 0x0 };

                    buf[i] = u8::from_i32(i32::from_le_bytes([
                        samples[i24_i],
                        samples[i24_i + 1],
                        samples[i24_i + 2],
                        sign_byte,
                    ]));
                    i += 1;
                }
                Ok(())
            } else if self.fmt.bit_depth == 32 {
                let mut samples = vec![0; buf.len() * 4];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from * 4) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    let i32_i = i * 4;
                    buf[i] = u8::from_f32(f32::from_le_bytes([
                        samples[i32_i],
                        samples[i32_i + 1],
                        samples[i32_i + 2],
                        samples[i32_i + 3],
                    ]));
                    i += 1;
                }
                Ok(())
            } else {
                return Err(ReadError::Parser(Error::UnsupportedFormat(
                    self.fmt.audio_format.to_u16(),
                )));
            }
        }

        /// Read 16-bit signed integer values from the WAV file from the start position (in number of samples * num_channels) to the number of values in the buffer. The start position plus the length of the buffer must be less than the total number of (samples * num_channels) in the WAV file.
        pub async fn read_i16(
            &mut self,
            from: usize,
            buf: &mut [i16],
        ) -> Result<(), ReadError<R::Error>> {
            if self.fmt.bit_depth == 8 {
                let mut samples = vec![0; buf.len()];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    buf[i] = i16::from_u8(samples[i]);
                    i += 1;
                }
                Ok(())
            } else if self.fmt.bit_depth == 16 {
                #[cfg(target_endian = "little")]
                {
                    // Optimized path for little-endian 16-bit reads
                    let bytes = unsafe {
                        core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 2)
                    };
                    self.reader
                        .seek(SeekFrom::Start((self.data_pos + from * 2) as u64))
                        .await
                        .map_err(|e| ReadError::Reader(e))?;
                    self.reader.read_exact(bytes).await?;
                    Ok(())
                }
                #[cfg(not(target_endian = "little"))]
                {
                    // Fallback for non-little-endian systems
                    let mut samples = vec![0; buf.len() * 2];
                    self.reader
                        .seek(SeekFrom::Start((self.data_pos + from * 2) as u64))
                        .await
                        .map_err(|e| ReadError::Reader(e))?;
                    self.reader.read_exact(&mut samples).await?;
                    let mut i = 0;
                    while i < buf.len() {
                        let i16_i = i * 2;
                        buf[i] = i16::from_le_bytes([samples[i16_i], samples[i16_i + 1]]);
                        i += 1;
                    }
                    Ok(())
                }
            } else if self.fmt.bit_depth == 24 {
                let mut samples = vec![0; buf.len() * 3];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from * 3) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    let i24_i = i * 3;
                    let sign = samples[i24_i + 2] >> 7;
                    let sign_byte = if sign == 1 { 0xff } else { 0x0 };

                    buf[i] = i16::from_i32(i32::from_le_bytes([
                        samples[i24_i],
                        samples[i24_i + 1],
                        samples[i24_i + 2],
                        sign_byte,
                    ]));
                    i += 1;
                }
                Ok(())
            } else if self.fmt.bit_depth == 32 {
                let mut samples = vec![0; buf.len() * 4];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from * 4) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    let i32_i = i * 4;
                    buf[i] = i16::from_f32(f32::from_le_bytes([
                        samples[i32_i],
                        samples[i32_i + 1],
                        samples[i32_i + 2],
                        samples[i32_i + 3],
                    ]));
                    i += 1;
                }
                Ok(())
            } else {
                return Err(ReadError::Parser(Error::UnsupportedFormat(
                    self.fmt.audio_format.to_u16(),
                )));
            }
        }

        /// Read 24-bit signed integer values from the WAV file from the start position (in number of samples * num_channels) to the number of values in the buffer. The start position plus the length of the buffer must be less than the total number of (samples * num_channels) in the WAV file.
        pub async fn read_i24(
            &mut self,
            from: usize,
            buf: &mut [i32],
        ) -> Result<(), ReadError<R::Error>> {
            if self.fmt.bit_depth == 8 {
                let mut samples = vec![0; buf.len()];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    buf[i] = i32::from_u8(samples[i]);
                    i += 1;
                }
                Ok(())
            } else if self.fmt.bit_depth == 16 {
                let mut samples = vec![0; buf.len() * 2];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from * 2) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    let i16_i = i * 2;
                    buf[i] =
                        i32::from_i16(i16::from_le_bytes([samples[i16_i], samples[i16_i + 1]]));
                    i += 1;
                }
                Ok(())
            } else if self.fmt.bit_depth == 24 {
                let mut samples = vec![0; buf.len() * 3];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from * 3) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    let sign = samples[i * 3 + 2] >> 7;
                    let sign_byte = if sign == 1 { 0xff } else { 0x0 };

                    let i32_sample = i32::from_le_bytes([
                        samples[i * 3],
                        samples[i * 3 + 1],
                        samples[i * 3 + 2],
                        sign_byte,
                    ]);
                    let i24_sample = i32::from_i32(i32_sample);
                    buf[i] = i24_sample;
                    i += 1;
                }
                Ok(())
            } else if self.fmt.bit_depth == 32 {
                let mut samples = vec![0; buf.len() * 4];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from * 4) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    let i32_i = i * 4;
                    buf[i] = i32::from_f32(f32::from_le_bytes([
                        samples[i32_i],
                        samples[i32_i + 1],
                        samples[i32_i + 2],
                        samples[i32_i + 3],
                    ]));
                    i += 1;
                }
                Ok(())
            } else {
                return Err(ReadError::Parser(Error::UnsupportedFormat(
                    self.fmt.audio_format.to_u16(),
                )));
            }
        }

        /// Read 32-bit floating point values from the WAV file from the start position (in number of samples * num_channels) to the number of values in the buffer. The start position plus the length of the buffer must be less than the total number of (samples * num_channels) in the WAV file.
        pub async fn read_f32(
            &mut self,
            from: usize,
            buf: &mut [f32],
        ) -> Result<(), ReadError<R::Error>> {
            if self.fmt.bit_depth == 8 {
                let mut samples = vec![0; buf.len()];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    buf[i] = f32::from_u8(samples[i]);
                    i += 1;
                }
                Ok(())
            } else if self.fmt.bit_depth == 16 {
                let mut samples = vec![0; buf.len() * 2];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from * 2) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    buf[i] =
                        f32::from_i16(i16::from_le_bytes([samples[i * 2], samples[i * 2 + 1]]));
                    i += 1;
                }
                Ok(())
            } else if self.fmt.bit_depth == 24 {
                let mut samples = vec![0; buf.len() * 3];
                self.reader
                    .seek(SeekFrom::Start((self.data_pos + from * 3) as u64))
                    .await
                    .map_err(|e| ReadError::Reader(e))?;
                self.reader.read_exact(&mut samples).await?;
                let mut i = 0;
                while i < buf.len() {
                    let i32_i = i * 3;
                    let sign = samples[i32_i + 2] >> 7;
                    let sign_byte = if sign == 1 { 0xff } else { 0x0 };

                    buf[i] = f32::from_i32(i32::from_le_bytes([
                        samples[i32_i],
                        samples[i32_i + 1],
                        samples[i32_i + 2],
                        sign_byte,
                    ]));
                    i += 1;
                }
                Ok(())
            } else if self.fmt.bit_depth == 32 {
                #[cfg(target_endian = "little")]
                {
                    // Optimized path for little-endian 32-bit float reads
                    let bytes = unsafe {
                        core::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, buf.len() * 4)
                    };
                    self.reader
                        .seek(SeekFrom::Start((self.data_pos + from * 4) as u64))
                        .await
                        .map_err(|e| ReadError::Reader(e))?;
                    self.reader.read_exact(bytes).await?;
                    Ok(())
                }
                #[cfg(not(target_endian = "little"))]
                {
                    // Fallback for non-little-endian systems
                    let mut samples = vec![0; buf.len() * 4];
                    self.reader
                        .seek(SeekFrom::Start((self.data_pos + from * 4) as u64))
                        .await
                        .map_err(|e| ReadError::Reader(e))?;
                    self.reader.read_exact(&mut samples).await?;
                    let mut i = 0;
                    while i < buf.len() {
                        let i32_i = i * 4;
                        buf[i] = f32::from_le_bytes([
                            samples[i32_i],
                            samples[i32_i + 1],
                            samples[i32_i + 2],
                            samples[i32_i + 3],
                        ]);
                        i += 1;
                    }
                    Ok(())
                }
            } else {
                return Err(ReadError::Parser(Error::UnsupportedFormat(
                    self.fmt.audio_format.to_u16(),
                )));
            }
        }
    }
}

//-----------------------------------
// MARK: Tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Data;
    use alloc::vec;

    #[test]
    fn test_incremental_wav_creation() {
        // Create a simple WAV file in memory
        let wav = crate::Wav::from_data(Data::BitDepth16(vec![1, 2, 3, -1]), 48_000, 2);
        let bytes = wav.to_bytes();

        // Create incremental WAV
        let incremental_wav = IncrementalWav::from_reader_default(&bytes[..]).unwrap();

        assert_eq!(incremental_wav.fmt.sample_rate, 48_000);
        assert_eq!(incremental_wav.fmt.num_channels, 2);
        assert_eq!(incremental_wav.fmt.bit_depth, 16);
    }

    #[test]
    fn test_incremental_reading() {
        // Create a WAV file with known samples
        let original_samples = vec![1, 2, 3, -1, 5, 6, 7, 8];
        let wav = crate::Wav::from_data(Data::BitDepth16(original_samples.clone()), 48_000, 2);
        let bytes = wav.to_bytes();

        // Create incremental WAV
        let incremental_wav = IncrementalWav::from_reader_default(&bytes[..]).unwrap();

        // Verify format information is correct
        assert_eq!(incremental_wav.fmt.sample_rate, 48_000);
        assert_eq!(incremental_wav.fmt.num_channels, 2);
        assert_eq!(incremental_wav.fmt.bit_depth, 16);

        // Create iterator for reading audio data
        let iterator = incremental_wav.iter_data::<i16>();

        // Read all samples and verify they match the original
        let mut samples_read = vec![];
        for result in iterator {
            match result {
                Ok(sample) => samples_read.push(sample),
                Err(e) => panic!("Error reading sample: {:?}", e),
            }
        }

        // Verify we got the expected number of samples
        assert_eq!(samples_read.len(), original_samples.len());

        // Verify the samples match the original data
        assert_eq!(samples_read, original_samples);

        // Test with f32 conversion
        let incremental_wav_f32 = IncrementalWav::from_reader_default(&bytes[..]).unwrap();
        let iterator_f32 = incremental_wav_f32.iter_data::<f32>();

        let mut f32_samples = vec![];
        for result in iterator_f32 {
            match result {
                Ok(sample) => f32_samples.push(sample),
                Err(e) => panic!("Error reading f32 sample: {:?}", e),
            }
        }

        // Verify f32 samples are normalized correctly
        assert_eq!(f32_samples.len(), original_samples.len());

        // Check that f32 samples are properly normalized
        let expected_f32: Vec<f32> = original_samples
            .iter()
            .map(|&s| s as f32 / 32768.0)
            .collect();

        assert_eq!(f32_samples, expected_f32);
    }

    #[test]
    fn test_8bit_audio() {
        // Test with 8-bit audio
        let original_samples = vec![128, 255, 0, 64, 192];
        let wav = crate::Wav::from_data(Data::BitDepth8(original_samples.clone()), 44_100, 1);
        let bytes = wav.to_bytes();

        let incremental_wav = IncrementalWav::from_reader_default(&bytes[..]).unwrap();
        assert_eq!(incremental_wav.fmt.bit_depth, 8);
        assert_eq!(incremental_wav.fmt.num_channels, 1);

        let iterator = incremental_wav.iter_data::<u8>();
        let mut samples_read = vec![];
        for result in iterator {
            match result {
                Ok(sample) => samples_read.push(sample),
                Err(e) => panic!("Error reading 8-bit sample: {:?}", e),
            }
        }

        assert_eq!(samples_read, original_samples);
    }

    #[test]
    fn test_24bit_audio() {
        // Test with 24-bit audio
        let original_samples = vec![1_000_000, -1_000_000, 500_000, -500_000];
        let wav = crate::Wav::from_data(Data::BitDepth24(original_samples.clone()), 96_000, 2);
        let bytes = wav.to_bytes();

        let incremental_wav = IncrementalWav::from_reader_default(&bytes[..]).unwrap();
        assert_eq!(incremental_wav.fmt.bit_depth, 24);
        assert_eq!(incremental_wav.fmt.num_channels, 2);

        let iterator = incremental_wav.iter_data::<i32>();
        let mut samples_read = vec![];
        for result in iterator {
            match result {
                Ok(sample) => samples_read.push(sample),
                Err(e) => panic!("Error reading 24-bit sample: {:?}", e),
            }
        }

        assert_eq!(samples_read, original_samples);
    }

    #[test]
    fn test_32bit_float_audio() {
        // Test with 32-bit float audio
        let original_samples = vec![0.0, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25];
        let wav = crate::Wav::from_data(Data::Float32(original_samples.clone()), 96_000, 2);
        let bytes = wav.to_bytes();

        let incremental_wav = IncrementalWav::from_reader_default(&bytes[..]).unwrap();
        assert_eq!(incremental_wav.fmt.bit_depth, 32);
        assert_eq!(incremental_wav.fmt.num_channels, 2);

        // Test reading as f32 (original format)
        let iterator = incremental_wav.iter_data::<f32>();
        let mut samples_read = vec![];
        for result in iterator {
            match result {
                Ok(sample) => samples_read.push(sample),
                Err(e) => panic!("Error reading 32-bit float sample: {:?}", e),
            }
        }

        assert_eq!(samples_read, original_samples);

        // Test reading as f64 (converted)
        let incremental_wav_f64 = IncrementalWav::from_reader_default(&bytes[..]).unwrap();
        let iterator_f64 = incremental_wav_f64.iter_data::<f64>();
        let mut f64_samples = vec![];
        for result in iterator_f64 {
            match result {
                Ok(sample) => f64_samples.push(sample),
                Err(e) => panic!("Error reading f64 sample: {:?}", e),
            }
        }

        let expected_f64: Vec<f64> = original_samples.iter().map(|&s| s as f64).collect();
        assert_eq!(f64_samples, expected_f64);

        // Test reading as i16 (converted)
        let incremental_wav_i16 = IncrementalWav::from_reader_default(&bytes[..]).unwrap();
        let iterator_i16 = incremental_wav_i16.iter_data::<i16>();
        let mut i16_samples = vec![];
        for result in iterator_i16 {
            match result {
                Ok(sample) => i16_samples.push(sample),
                Err(e) => panic!("Error reading i16 sample: {:?}", e),
            }
        }

        let expected_i16: Vec<i16> = original_samples
            .iter()
            .map(|&s| (s * 32767.5).clamp(-32767.5, 32767.5) as i16)
            .collect();
        assert_eq!(i16_samples, expected_i16);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_with_real_files() {
        let files_16 = [
            "./test_files/mono_16_48000.wav",
            "./test_files/stereo_16_48000.wav",
        ];
        let files_24 = [
            "./test_files/mono_24_48000.wav",
            "./test_files/stereo_24_48000.wav",
        ];

        for file in files_16 {
            // Test IncrementalWav creation
            let path = std::path::Path::new(file);
            let incremental_wav = IncrementalWav::from_file(path).unwrap();

            // Verify format information is correct
            assert!(incremental_wav.fmt.sample_rate > 0);
            assert!(incremental_wav.fmt.num_channels > 0);
            assert!(incremental_wav.fmt.bit_depth > 0);

            // Test that we can read samples
            let iterator = incremental_wav.iter_data::<i16>();
            let mut samples = vec![];
            for result in iterator {
                match result {
                    Ok(_sample) => {
                        samples.push(_sample);
                    }
                    Err(e) => panic!("Error reading sample from {}: {:?}", file, e),
                }
            }

            // Verify we got some samples
            assert!(samples.len() > 0, "No samples read from {}", file);

            // Parse with regular Wav
            let bytes = std::fs::read(path).unwrap();
            let wav = crate::Wav::from_bytes(&bytes).unwrap();
            match wav.data {
                Data::BitDepth16(regular_samples) => {
                    assert_eq!(samples.len(), regular_samples.len());
                    assert_eq!(samples, regular_samples);
                }
                _ => panic!("Expected 16-bit audio"),
            }
        }

        for file in files_24 {
            // Test IncrementalWav creation
            let path = std::path::Path::new(file);
            let incremental_wav = IncrementalWav::from_file(path).unwrap();

            // Verify format information is correct
            assert!(incremental_wav.fmt.sample_rate > 0);
            assert!(incremental_wav.fmt.num_channels > 0);
            assert!(incremental_wav.fmt.bit_depth > 0);

            // Test that we can read samples
            let iterator = incremental_wav.iter_data::<i32>();
            let mut samples = vec![];
            for result in iterator {
                match result {
                    Ok(_sample) => {
                        samples.push(_sample);
                    }
                    Err(e) => panic!("Error reading sample from {}: {:?}", file, e),
                }
            }

            // Verify we got some samples
            assert!(samples.len() > 0, "No samples read from {}", file);

            // Parse with regular Wav
            let bytes = std::fs::read(path).unwrap();

            let wav = crate::Wav::from_bytes(&bytes).unwrap();
            match wav.data {
                Data::BitDepth24(regular_samples) => {
                    assert_eq!(samples.len(), regular_samples.len());
                    assert_eq!(samples, regular_samples);
                }
                _ => panic!("Expected 24-bit audio"),
            }
        }
    }

    #[cfg(test)]
    mod async_tests {
        use super::*;
        use crate::data::Data;
        use alloc::vec;

        // Simple async reader wrapper for testing
        struct AsyncSliceReader<'a> {
            data: &'a [u8],
            pos: usize,
        }

        impl<'a> AsyncSliceReader<'a> {
            fn new(data: &'a [u8]) -> Self {
                Self { data, pos: 0 }
            }
        }

        impl embedded_io_async::ErrorType for AsyncSliceReader<'_> {
            type Error = std::io::Error;
        }

        impl embedded_io_async::Read for AsyncSliceReader<'_> {
            async fn read(&mut self, buf: &mut [u8]) -> Result<usize, Self::Error> {
                if self.pos >= self.data.len() {
                    return Ok(0);
                }
                let remaining = self.data.len() - self.pos;
                let to_read = remaining.min(buf.len());
                buf[..to_read].copy_from_slice(&self.data[self.pos..self.pos + to_read]);
                self.pos += to_read;
                Ok(to_read)
            }
        }

        impl embedded_io_async::Seek for AsyncSliceReader<'_> {
            async fn seek(&mut self, pos: embedded_io_async::SeekFrom) -> Result<u64, Self::Error> {
                let new_pos = match pos {
                    embedded_io_async::SeekFrom::Start(offset) => offset as usize,
                    embedded_io_async::SeekFrom::Current(offset) => {
                        if offset >= 0 {
                            self.pos + offset as usize
                        } else {
                            self.pos.saturating_sub((-offset) as usize)
                        }
                    }
                    embedded_io_async::SeekFrom::End(offset) => {
                        if offset >= 0 {
                            self.data.len() + offset as usize
                        } else {
                            self.data.len().saturating_sub((-offset) as usize)
                        }
                    }
                };
                self.pos = new_pos.min(self.data.len());
                Ok(self.pos as u64)
            }
        }

        #[tokio::test]
        async fn test_async_incremental_wav_creation() {
            // Create a simple WAV file in memory
            let wav = crate::Wav::from_data(Data::BitDepth16(vec![1, 2, 3, -1]), 48_000, 2);
            let bytes = wav.to_bytes();

            // Create async incremental WAV
            let reader = AsyncSliceReader::new(&bytes);
            let incremental_wav = asynch::IncrementalWav::from_reader(reader).await.unwrap();

            assert_eq!(incremental_wav.fmt.sample_rate, 48_000);
            assert_eq!(incremental_wav.fmt.num_channels, 2);
            assert_eq!(incremental_wav.fmt.bit_depth, 16);
        }

        #[tokio::test]
        async fn test_async_read_u8() {
            // Test with 16-bit source
            let original_samples = vec![1, 2, 3, -1, 5, 6, 7, 8];
            let wav = crate::Wav::from_data(Data::BitDepth16(original_samples.clone()), 48_000, 2);
            let bytes = wav.to_bytes();

            let reader = AsyncSliceReader::new(&bytes);
            let mut incremental_wav = asynch::IncrementalWav::from_reader(reader).await.unwrap();

            // Read as u8
            let mut buf = vec![0u8; incremental_wav.num_sample_values()];
            incremental_wav.read_u8(0, &mut buf).await.unwrap();

            // The async function reads raw bytes and converts using FromWavSample
            // Let's verify this matches what we get from the regular WAV parsing
            let regular_wav = crate::Wav::from_bytes(&bytes).unwrap();
            let expected_u8: Vec<u8> = regular_wav.iter_as::<u8>().collect();

            assert_eq!(buf, expected_u8);
        }

        #[tokio::test]
        async fn test_async_read_i16() {
            // Test with 16-bit source (optimized path)
            let original_samples = vec![1, 2, 3, -1, 5, 6, 7, 8];
            let wav = crate::Wav::from_data(Data::BitDepth16(original_samples.clone()), 48_000, 2);
            let bytes = wav.to_bytes();

            let reader = AsyncSliceReader::new(&bytes);
            let mut incremental_wav = asynch::IncrementalWav::from_reader(reader).await.unwrap();

            // Read as i16
            let mut buf = vec![0i16; original_samples.len()];
            incremental_wav.read_i16(0, &mut buf).await.unwrap();

            assert_eq!(buf, original_samples);

            // Test with 8-bit source
            let original_8bit = vec![128u8, 255, 0, 64, 192];
            let wav_8bit = crate::Wav::from_data(Data::BitDepth8(original_8bit.clone()), 44_100, 1);
            let bytes_8bit = wav_8bit.to_bytes();

            let reader_8bit = AsyncSliceReader::new(&bytes_8bit);
            let mut incremental_wav_8bit = asynch::IncrementalWav::from_reader(reader_8bit)
                .await
                .unwrap();

            let mut buf_8bit = vec![0i16; original_8bit.len()];
            incremental_wav_8bit
                .read_i16(0, &mut buf_8bit)
                .await
                .unwrap();

            let expected_i16: Vec<i16> = original_8bit.iter().map(|&s| i16::from_u8(s)).collect();
            assert_eq!(buf_8bit, expected_i16);
        }

        #[tokio::test]
        async fn test_async_read_i24() {
            // Test with 24-bit source
            let original_samples = vec![1_000_000, -1_000_000, 500_000, -500_000];
            let wav = crate::Wav::from_data(Data::BitDepth24(original_samples.clone()), 96_000, 2);
            let bytes = wav.to_bytes();

            let reader = AsyncSliceReader::new(&bytes);
            let mut incremental_wav = asynch::IncrementalWav::from_reader(reader).await.unwrap();

            // Read as i32 (24-bit stored as i32)
            let mut buf = vec![0i32; original_samples.len()];
            incremental_wav.read_i24(0, &mut buf).await.unwrap();

            // The async function should read the raw 24-bit data correctly
            // Let's verify this matches what we get from the regular WAV parsing
            let regular_wav = crate::Wav::from_bytes(&bytes).unwrap();
            let expected_i32: Vec<i32> = regular_wav.iter_as::<i32>().collect();
            assert_eq!(buf, expected_i32);

            // Test with 16-bit source
            let original_16bit = vec![1i16, 2, 3, -1, 5];
            let wav_16bit =
                crate::Wav::from_data(Data::BitDepth16(original_16bit.clone()), 48_000, 2);
            let bytes_16bit = wav_16bit.to_bytes();

            let reader_16bit = AsyncSliceReader::new(&bytes_16bit);
            let mut incremental_wav_16bit = asynch::IncrementalWav::from_reader(reader_16bit)
                .await
                .unwrap();

            let mut buf_16bit = vec![0i32; original_16bit.len()];
            incremental_wav_16bit
                .read_i24(0, &mut buf_16bit)
                .await
                .unwrap();

            let expected_i32_16bit: Vec<i32> =
                original_16bit.iter().map(|&s| i32::from_i16(s)).collect();
            assert_eq!(buf_16bit, expected_i32_16bit);
        }

        #[tokio::test]
        async fn test_async_read_f32() {
            // Test with 32-bit float source (optimized path)
            let original_samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0, 0.25, -0.25];
            let wav = crate::Wav::from_data(Data::Float32(original_samples.clone()), 96_000, 2);
            let bytes = wav.to_bytes();

            let reader = AsyncSliceReader::new(&bytes);
            let mut incremental_wav = asynch::IncrementalWav::from_reader(reader).await.unwrap();

            // Read as f32
            let mut buf = vec![0.0f32; original_samples.len()];
            incremental_wav.read_f32(0, &mut buf).await.unwrap();

            assert_eq!(buf, original_samples);

            // Test with 16-bit source
            let original_16bit = vec![1i16, 2, 3, -1, 5];
            let wav_16bit =
                crate::Wav::from_data(Data::BitDepth16(original_16bit.clone()), 48_000, 2);
            let bytes_16bit = wav_16bit.to_bytes();

            let reader_16bit = AsyncSliceReader::new(&bytes_16bit);
            let mut incremental_wav_16bit = asynch::IncrementalWav::from_reader(reader_16bit)
                .await
                .unwrap();

            let mut buf_16bit = vec![0.0f32; original_16bit.len()];
            incremental_wav_16bit
                .read_f32(0, &mut buf_16bit)
                .await
                .unwrap();

            let expected_f32: Vec<f32> = original_16bit.iter().map(|&s| f32::from_i16(s)).collect();
            assert_eq!(buf_16bit, expected_f32);
        }

        #[tokio::test]
        async fn test_async_partial_reads() {
            // Test reading partial data from different positions
            let original_samples = vec![1i16, 2, 3, -1, 5, 6, 7, 8, 9, 10];
            let wav = crate::Wav::from_data(Data::BitDepth16(original_samples.clone()), 48_000, 2);
            let bytes = wav.to_bytes();

            let reader = AsyncSliceReader::new(&bytes);
            let mut incremental_wav = asynch::IncrementalWav::from_reader(reader).await.unwrap();

            // Read first 3 samples
            let mut buf1 = vec![0i16; 3];
            incremental_wav.read_i16(0, &mut buf1).await.unwrap();
            assert_eq!(buf1, vec![1, 2, 3]);

            // Read next 3 samples starting from position 3
            let mut buf2 = vec![0i16; 3];
            incremental_wav.read_i16(3, &mut buf2).await.unwrap();
            assert_eq!(buf2, vec![-1, 5, 6]);

            // Read last 4 samples starting from position 6
            let mut buf3 = vec![0i16; 4];
            incremental_wav.read_i16(6, &mut buf3).await.unwrap();
            assert_eq!(buf3, vec![7, 8, 9, 10]);
        }

        #[tokio::test]
        async fn test_async_num_sample_values() {
            let original_samples = vec![1i16, 2, 3, -1, 5, 6, 7, 8];
            let wav = crate::Wav::from_data(Data::BitDepth16(original_samples.clone()), 48_000, 2);
            let bytes = wav.to_bytes();

            let reader = AsyncSliceReader::new(&bytes);
            let incremental_wav = asynch::IncrementalWav::from_reader(reader).await.unwrap();

            // For 16-bit audio, each sample is 2 bytes
            // data_len should be the total number of bytes in the data chunk
            // num_sample_values should be data_len / bytes_per_sample
            assert_eq!(incremental_wav.num_sample_values(), original_samples.len());
        }
    }
}
