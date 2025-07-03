//! Basic `no_std` library for parsing and creating WAV files.
//!
//! Reading a WAV file:
//! ```
//! use std::fs;
//! use std::path::Path;
//! use wavv::{Wav, Data};
//!
//! fn main() {
//!     let bytes = fs::read(Path::new("./test_files/stereo_16_48000.wav")).unwrap();
//! 	let wav = Wav::from_bytes(&bytes).unwrap();
//!
//!     assert_eq!(wav.fmt.num_channels, 2);
//!     assert_eq!(wav.fmt.bit_depth, 16);
//!     assert_eq!(wav.fmt.sample_rate, 48_000);
//!
//!     match wav.data {
//!         Data::BitDepth8(samples) => println!("{:?}", samples),
//!         Data::BitDepth16(samples) => println!("{:?}", samples),
//!         Data::BitDepth24(samples) => println!("{:?}", samples),
//!         Data::Float32(samples) => println!("{:?}", samples),
//!     }
//! }
//! ```
//!
//! Using the generic iterator to convert samples to different types:
//! ```
//! use wavv::{Wav, Data};
//!
//! let wav = Wav::from_data(Data::BitDepth16(vec![1, 2, 3, -1]), 48_000, 2);
//!
//! // Iterate as normalized f32 samples ([-1.0, 1.0])
//! let f32_samples: Vec<f32> = wav.iter_as::<f32>().collect();
//!
//! // Iterate as normalized f64 samples ([-1.0, 1.0])
//! let f64_samples: Vec<f64> = wav.iter_as::<f64>().collect();
//!
//! // Iterate as i32 samples (preserving original values)
//! let i32_samples: Vec<i32> = wav.iter_as::<i32>().collect();
//!
//! // You can also work with 32-bit float WAV files directly
//! let wav_float = Wav::from_data(Data::Float32(vec![0.5, -0.5, 1.0, -1.0]), 48_000, 2);
//! let float_samples: Vec<f32> = wav_float.iter_as::<f32>().collect();
//! ```
//!
//! Incremental reading with PartialWav (requires "embedded" feature):
//! ```
//! use std::fs;
//!
//! #[cfg(feature = "embedded")]
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!
//!     let file_bytes = fs::read("./test_files/stereo_16_48000.wav")?;
//!
//!     // Only read the format information initially
//!     let partial_wav = wavv::PartialWav::from_reader_default(&file_bytes[..]).unwrap();
//!
//!     println!("Sample rate: {}", partial_wav.fmt.sample_rate);
//!     println!("Channels: {}", partial_wav.fmt.num_channels);
//!     println!("Bit depth: {}", partial_wav.fmt.bit_depth);
//!
//!     // Now read audio data incrementally
//!     let mut samples = vec![];
//!
//!     for result in partial_wav.iter_data::<f32>() {
//!         match result {
//!             Ok(sample) => samples.push(sample),
//!             Err(e) => panic!("Error: {:?}", e),
//!         }
//!     }
//!
//!     Ok(())
//! }
//!
//! # #[cfg(not(feature = "embedded"))]
//! # fn main() {}
//! ```
//!
//! Writing a WAV file:
//! ```
//! use std::fs::File;
//! use std::io::Write;
//! use std::path::Path;
//! use wavv::{Wav, Data};
//!
//! fn main() {
//!     // Enjoy the silence
//!     let data = Data::Float32(vec![0.0; 480_000]);
//! 	let wav = Wav::from_data(data, 48_000, 2);
//!
//!     let path = Path::new("output.wav");
//!     let mut file = File::create(&path).unwrap();
//!     file.write_all(&wav.to_bytes()).unwrap();
//! }
//! ```

#![cfg_attr(all(not(test), not(feature = "std")), no_std)]
#![warn(missing_docs)]

extern crate alloc;

mod chunk;
mod conversion;
mod data;
mod error;
mod fmt;
mod wav;

pub use chunk::{Chunk, ChunkTag};
pub use conversion::FromWavSample;
pub use data::Data;
pub use error::Error;
pub use fmt::{AudioFormat, Fmt};
pub use wav::{Wav, WavIterator};

#[cfg(feature = "embedded")]
mod partial;
#[cfg(feature = "embedded")]
pub use partial::{PartialWav, PartialWavIterator};
