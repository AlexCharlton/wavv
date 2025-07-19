# wavv
[![.github/workflows/main.yml](https://github.com/samuelleeuwenburg/wavv/actions/workflows/main.yml/badge.svg)](https://github.com/samuelleeuwenburg/wavv/actions/workflows/main.yml)
[![Crates.io](https://img.shields.io/crates/v/wavv.svg)](https://crates.io/crates/wavv)
[![docs.rs](https://docs.rs/wavv/badge.svg)](https://docs.rs/wavv/)


Basic `no_std` library for parsing and creating WAV files.

Reading a WAV file:
```
use std::fs;
use std::path::Path;
use wavv::{Wav, Data};

fn main() {
    let bytes = fs::read(Path::new("./test_files/stereo_16_48000.wav")).unwrap();
	let wav = Wav::from_bytes(&bytes).unwrap();

    assert_eq!(wav.fmt.num_channels, 2);
    assert_eq!(wav.fmt.bit_depth, 16);
    assert_eq!(wav.fmt.sample_rate, 48_000);

    match wav.data {
        Data::BitDepth8(samples) => println!("{:?}", samples),
        Data::BitDepth16(samples) => println!("{:?}", samples),
        Data::BitDepth24(samples) => println!("{:?}", samples),
        Data::Float32(samples) => println!("{:?}", samples),
    }
}
```

Writing a WAV file:
```
use std::fs::File;
use std::io::Write;
use std::path::Path;
use wavv::{Wav, Data};

fn main() {
    let data = Data::BitDepth16(vec![0, 0, 0, 0, 0, 0]);
	let wav = Wav::from_data(data, 48_000, 2);

    let path = Path::new("output.wav");
    let mut file = File::create(&path).unwrap();
    file.write_all(&wav.to_bytes()).unwrap();
}
```

## Finalizing this new API
- Provide `read_*` along with `read_*_from` methods
- Line up the sync version with the async (drop the iterator)
- Create `incremental` and `incremental_async` features
- When `std`, don't use embedded_io traits, use the std ones
  - Implies a `no_std` trait
  - Or maybe: `incremental_std`, `incremental_async_std`, `incremental_no_std` and `incremental_async_no_std`
  - Use https://crates.io/crates/dasp_sample ?