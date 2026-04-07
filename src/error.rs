use std::{fmt, io};

pub type Result<T> = core::result::Result<T, Error>;

#[derive(Debug)]
pub enum Error {
    IO(io::Error),
    FileTooSmall,
    InvalidVersion(i32),
    InvalidMetaValueType(u32),
    InvalidGGMLType(u32),
    UnsupportedArrayValue,
    UnsupportedFileFormat(i32),
    TensorWriteIndex(usize),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::IO(e) => write!(f, "{e}"),
            Error::FileTooSmall => write!(f, "gguf file too small to be valid"),
            Error::InvalidVersion(v) => write!(f, "invalid version {v}, only supports versions: 1 | 2 | 3"),
            Error::InvalidMetaValueType(t) => write!(f, "invalid metadata value type {t}"),
            Error::InvalidGGMLType(t) => write!(f, "invalid ggml type {t}"),
            Error::UnsupportedArrayValue => write!(f, "unsupported item value type: Array"),
            Error::UnsupportedFileFormat(magic) => write!(f, "unsupported file format {}", fmt_magic(magic)),
            Error::TensorWriteIndex(_) => write!(f, "tensor index is out of bounds"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::IO(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::IO(e)
    }
}

fn fmt_magic(val: &i32) -> String {
    let bytes = val.to_be_bytes();
    let ascii: String = bytes.iter().map(|b| *b as char).collect();
    format!("{ascii} (0x{:08x})", *val)
}
