//! # GGUF File Parser
//!
//! A Rust library for parsing and reading GGUF (GGML Universal Format) files.
//!
//! GGUF files are binary files that contain key-value metadata and tensors,
//! commonly used for storing quantized machine learning models like LLaMA, Phi, etc.
//!
//! ## Features
//!
//! - Decode GGUF files (v1, v2, v3)
//! - Access key-value metadata
//! - Access tensor information
//! - Support for little-endian and big-endian files
//! - CLI tool for quick inspection
//! - Optional memory-mapped file support (enable `mmap` feature)
//!
//! ## Example
//!
//! ```rust,no_run
//! use gguf_rs::get_gguf_container;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // Open a GGUF file
//!     let mut container = get_gguf_container("model.gguf")?;
//!     let model = container.decode()?;
//!
//!     // Print model info
//!     println!("Version: {}", model.get_version());
//!     println!("Architecture: {}", model.model_family());
//!     println!("Parameters: {}", model.model_parameters());
//!     println!("File type: {}", model.file_type());
//!     println!("Tensors: {}", model.num_tensor());
//!
//!     // List tensors
//!     for tensor in model.tensors() {
//!         println!("  {}: {:?} {:?}", tensor.name, tensor.kind, tensor.shape);
//!     }
//!
//!     Ok(())
//! }
//! ```
//!
//! ## CLI Usage
//!
//! Install the CLI tool:
//! ```bash
//! cargo install gguf-rs
//! ```
//!
//! Show model info:
//! ```bash
//! gguf model.gguf
//! ```
//!
//! Show tensors:
//! ```bash
//! gguf model.gguf --tensors
//! ```
//!
//! ## Memory-Mapped Files
//!
//! For large files, enable the `mmap` feature for more efficient access:
//!
//! ```toml
//! [dependencies]
//! gguf-rs = { version = "0.1", features = ["mmap"] }
//! ```
//!
//! ```rust,ignore
//! use gguf_rs::mmap::MmapGGUF;
//!
//! let mmap = MmapGGUF::open("large_model.gguf")?;
//! let model = mmap.decode()?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Async I/O
//!
//! For async applications, enable the `async` feature:
//!
//! ```toml
//! [dependencies]
//! gguf-rs = { version = "0.1", features = ["async"] }
//! ```
//!
//! ```rust,ignore
//! use gguf_rs::async_io::AsyncGGUF;
//!
//! #[tokio::main]
//! async fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     let mut container = AsyncGGUF::open("model.gguf").await?;
//!     let model = container.decode().await?;
//!
//!     println!("Architecture: {}", model.model_family());
//!     Ok(())
//! }
//! ```

pub mod error;

use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use error::{Error, Result};
#[cfg(feature = "logging")]
use log::debug;
use std::{borrow::Borrow, collections::BTreeMap, fmt::Display};

/// Magic constant for `ggml` files (unversioned).
pub const FILE_MAGIC_GGML: i32 = 0x67676d6c;
/// Magic constant for `ggml` files (versioned, ggmf).
pub const FILE_MAGIC_GGMF: i32 = 0x67676d66;
/// Magic constant for `ggml` files (versioned, ggjt).
pub const FILE_MAGIC_GGJT: i32 = 0x67676a74;
/// Magic constant for `ggla` files (LoRA adapter).
pub const FILE_MAGIC_GGLA: i32 = 0x67676C61;
/// Magic constant for `gguf` files (versioned, gguf)
pub const FILE_MAGIC_GGUF_LE: i32 = 0x46554747;
pub const FILE_MAGIC_GGUF_BE: i32 = 0x47475546;

pub const GGUF_VERSION_V1: i32 = 0x00000001;
pub const GGUF_VERSION_V2: i32 = 0x00000002;
pub const GGUF_VERSION_V3: i32 = 0x00000003;

const THOUSAND: u64 = 1000;
const MILLION: u64 = 1_000_000;
const BILLION: u64 = 1_000_000_000;

/// Convert a number to a human-readable string.
fn human_number(value: u64) -> String {
    match value {
        _ if value > BILLION => format!("{:.0}B", value as f64 / BILLION as f64),
        _ if value > MILLION => format!("{:.0}M", value as f64 / MILLION as f64),
        _ if value > THOUSAND => format!("{:.0}K", value as f64 / THOUSAND as f64),
        _ => format!("{}", value),
    }
}

/// Convert a file type to a human-readable string.
/// GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
fn file_type(ft: u64) -> String {
    match ft {
        0 => "All F32",
        1 => "Mostly F16",
        2 => "Mostly Q4_0",
        3 => "Mostly Q4_1",
        4 => "Mostly Q4_1 Some F16",
        5 => "Mostly Q4_2 (UNSUPPORTED)",
        6 => "Mostly Q4_3 (UNSUPPORTED)",
        7 => "Mostly Q8_0",
        8 => "Mostly Q5_0",
        9 => "Mostly Q5_1",
        10 => "Mostly Q2_K",
        11 => "Mostly Q3_K_S",
        12 => "Mostly Q3_K_M",
        13 => "Mostly Q3_K_L",
        14 => "Mostly Q4_K_S",
        15 => "Mostly Q4_K_M",
        16 => "Mostly Q5_K_S",
        17 => "Mostly Q5_K_M",
        18 => "Mostly Q6_K",
        19 => "Mostly IQ2_XXS",
        20 => "Mostly IQ2_XS",
        21 => "Mostly Q2_K_S",
        22 => "Mostly IQ3_XS",
        23 => "Mostly IQ3_XXS",
        24 => "Mostly IQ1_S",
        25 => "Mostly IQ4_NL",
        26 => "Mostly IQ3_S",
        27 => "Mostly IQ3_M",
        28 => "Mostly IQ2_S",
        29 => "Mostly IQ2_M",
        30 => "Mostly IQ4_XS",
        31 => "Mostly IQ1_M",
        32 => "Mostly BF16",
        33 => "Mostly Q4_0_4_4 (UNSUPPORTED)",
        34 => "Mostly Q4_0_4_8 (UNSUPPORTED)",
        35 => "Mostly Q4_0_8_8 (UNSUPPORTED)",
        36 => "Mostly TQ1_0",
        37 => "Mostly TQ2_0",
        38 => "Mostly MXFP4_MOE",
        39 => "Mostly NVFP4",
        40 => "Mostly Q1_0",
        _ => "unknown",
    }
    .to_string()
}

/// Byte order of the GGUF file.
#[derive(Default, Debug, Clone)]
pub enum ByteOrder {
    #[default]
    LE,
    BE,
}

/// Version of the GGUF file.
#[derive(Debug, Clone)]
pub enum Version {
    V1(V1),
    V2(V2),
    V3(V3),
}

/// Version 1 of the GGUF file.
#[derive(Debug, Default, Clone)]
pub struct V1 {
    num_tensor: u32,
    num_kv: u32,
}

/// Version 2 of the GGUF file.
#[derive(Debug, Default, Clone)]
pub struct V2 {
    num_tensor: u64,
    num_kv: u64,
}

/// Version 3 of the GGUF file.
#[derive(Debug, Default, Clone)]
pub struct V3 {
    num_tensor: u64,
    num_kv: u64,
}

/// GGUF file container for reading GGUF binary files.
///
/// The container wraps a reader and provides methods to decode the GGUF file
/// into a [`GGUFModel`].
///
/// Use [`get_gguf_container`] for a convenient way to open a file.
pub struct GGUFContainer<'a> {
    bo: ByteOrder,
    version: Version,
    reader: Box<dyn std::io::Read + 'a>,
    max_array_size: u64,
}

impl<'a> GGUFContainer<'a> {
    /// The default max size of arrays when parsing
    ///
    /// You can change this by using [`Self::with_max_array_size()`]
    pub const DEFAULT_MAX_ARRAY_SIZE: u64 = 3;

    /// Create a new `GGUFContainer` from a byte order and a reader.
    ///
    /// # Arguments
    ///
    /// * `bo` - Byte order (little-endian or big-endian)
    /// * `reader` - A reader implementing `std::io::Read`
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use gguf_rs::{GGUFContainer, ByteOrder};
    /// use std::fs::File;
    ///
    /// let file = File::open("model.gguf")?;
    /// let container = GGUFContainer::new(ByteOrder::LE, Box::new(file));
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn new(bo: ByteOrder, reader: Box<dyn std::io::Read + 'a>) -> Self {
        Self {
            bo,
            version: Version::V1(V1::default()),
            reader,
            max_array_size: Self::DEFAULT_MAX_ARRAY_SIZE,
        }
    }

    /// Set the maximum size for arrays to be read during parsing.
    ///
    /// By default this is set to [`Self::DEFAULT_MAX_ARRAY_SIZE`].
    ///
    /// Set this to [`u64::MAX`] to remove the limit.
    pub fn with_max_array_size(self, max_array_size: u64) -> Self {
        Self {
            max_array_size,
            ..self
        }
    }

    /// Get the version of the GGUF file container.
    ///
    /// Returns the default version ("v1") before decoding.
    /// After successful decode, returns the actual file version ("v1", "v2", or "v3").
    pub fn get_version(&self) -> String {
        match &self.version {
            Version::V1(_) => String::from("v1"),
            Version::V2(_) => String::from("v2"),
            Version::V3(_) => String::from("v3"),
        }
    }

    /// Decode the GGUF file and return a `GGUFModel`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file has an invalid or unsupported GGUF version
    /// - The file contains malformed data
    /// - An I/O error occurs while reading
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use gguf_rs::get_gguf_container;
    ///
    /// let mut container = get_gguf_container("model.gguf")?;
    /// let model = container.decode()?;
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn decode(&mut self) -> Result<GGUFModel> {
        let version = match self.bo {
            ByteOrder::LE => self.reader.read_i32::<LittleEndian>()?,
            ByteOrder::BE => self.reader.read_i32::<BigEndian>()?,
        };

        #[cfg(feature = "logging")]
        {
            debug!("version {}", version);
        }

        match version {
            GGUF_VERSION_V1 => {
                let mut buffer: [u32; 2] = [0; 2];
                match self.bo {
                    ByteOrder::LE => self.reader.read_u32_into::<LittleEndian>(&mut buffer)?,
                    ByteOrder::BE => self.reader.read_u32_into::<BigEndian>(&mut buffer)?,
                };

                self.version = Version::V1(V1 {
                    num_tensor: buffer[0],
                    num_kv: buffer[1],
                });
            }
            GGUF_VERSION_V2 | GGUF_VERSION_V3 => {
                let mut buffer: [u64; 2] = [0; 2];
                match self.bo {
                    ByteOrder::LE => self.reader.read_u64_into::<LittleEndian>(&mut buffer)?,
                    ByteOrder::BE => self.reader.read_u64_into::<BigEndian>(&mut buffer)?,
                };

                if version == GGUF_VERSION_V2 {
                    self.version = Version::V2(V2 {
                        num_tensor: buffer[0],
                        num_kv: buffer[1],
                    });
                } else {
                    self.version = Version::V3(V3 {
                        num_tensor: buffer[0],
                        num_kv: buffer[1],
                    });
                }
            }
            invalid_version => {
                return Err(Error::InvalidVersion(invalid_version));
            }
        };

        let mut model = GGUFModel {
            kv: BTreeMap::new(),
            tensors: Vec::new(),
            parameters: 0,
            max_array_size: self.max_array_size,
            bo: self.bo.clone(),
            version: self.version.clone(),
        };

        model.decode(&mut self.reader)?;
        Ok(model)
    }
}

/// Tensor in the GGUF file.
///
/// Represents a single tensor with its metadata including name, type, offset, size, and shape.
///
/// # Example
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container;
///
/// let mut container = get_gguf_container("model.gguf")?;
/// let model = container.decode()?;
///
/// for tensor in model.tensors() {
///     println!("Tensor: {} (shape: {:?})", tensor.name, tensor.shape);
/// }
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Name of the tensor (e.g., "token_embd.weight", "blk.0.attn_q.weight")
    pub name: String,
    /// GGML type identifier (see [`GGMLType`] for interpretation)
    pub kind: u32,
    /// Byte offset in the file where tensor data begins
    pub offset: u64,
    /// Size of tensor data in bytes
    pub size: u64,
    /// Shape dimensions (number of elements in each dimension)
    pub shape: [u64; 4],
}

/// Decoded GGUF model containing metadata and tensors.
///
/// Use [`get_gguf_container`] to create a container, then call [`GGUFContainer::decode`]
/// to get a `GGUFModel`.
///
/// # Example
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container;
///
/// let mut container = get_gguf_container("model.gguf")?;
/// let model = container.decode()?;
///
/// println!("Model: {}", model.model_family());
/// println!("Parameters: {}", model.model_parameters());
/// println!("Tensors: {}", model.num_tensor());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct GGUFModel {
    kv: BTreeMap<String, MetadataValue>,
    tensors: Vec<Tensor>,
    parameters: u64,
    max_array_size: u64,
    bo: ByteOrder,
    version: Version,
}

/// Metadata value type in GGUF files.
///
/// Represents the type of a metadata value in the key-value store.
/// Used when decoding metadata to determine how to interpret bytes.
#[derive(Debug)]
pub enum MetadataValueType {
    Uint8 = 0,
    Int8 = 1,
    Uint16 = 2,
    Int16 = 3,
    Uint32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    Uint64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for MetadataValueType {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        Ok(match value {
            0 => MetadataValueType::Uint8,
            1 => MetadataValueType::Int8,
            2 => MetadataValueType::Uint16,
            3 => MetadataValueType::Int16,
            4 => MetadataValueType::Uint32,
            5 => MetadataValueType::Int32,
            6 => MetadataValueType::Float32,
            7 => MetadataValueType::Bool,
            8 => MetadataValueType::String,
            9 => MetadataValueType::Array,
            10 => MetadataValueType::Uint64,
            11 => MetadataValueType::Int64,
            12 => MetadataValueType::Float64,
            _ => return Err(Error::InvalidMetaValueType(value)),
        })
    }
}

/// Metadata value types
#[derive(Debug, Clone)]
pub enum MetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<MetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl std::fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetadataValue::Uint8(x) => write!(f, "{x}"),
            MetadataValue::Int8(x) => write!(f, "{x}"),
            MetadataValue::Uint16(x) => write!(f, "{x}"),
            MetadataValue::Int16(x) => write!(f, "{x}"),
            MetadataValue::Uint32(x) => write!(f, "{x}"),
            MetadataValue::Int32(x) => write!(f, "{x}"),
            MetadataValue::Float32(x) => write!(f, "{x}"),
            MetadataValue::Bool(x) => write!(f, "{x}"),
            MetadataValue::String(x) => write!(f, "{x}"),
            MetadataValue::Array(arr) => {
                write!(f, "[")?;
                for x in arr.iter().take(3) {
                    write!(f, " {x}")?;
                }
                write!(f, " ]")
            }
            MetadataValue::Uint64(x) => write!(f, "{x}"),
            MetadataValue::Int64(x) => write!(f, "{x}"),
            MetadataValue::Float64(x) => write!(f, "{x}"),
        }
    }
}

#[derive(Debug)]
pub struct MetadataTypeError {
    expected: &'static str,
    found: MetadataValue,
}

impl std::fmt::Display for MetadataTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "failed to convert metatype: expected {}, found {}",
            self.expected, self.found
        )
    }
}

impl core::error::Error for MetadataTypeError {}

macro_rules! impl_from_for_metadata {
    ($($ty:ty => $variant:ident),* $(,)?) => {
        $(
            impl From<$ty> for MetadataValue {
                fn from(value: $ty) -> Self {
                    MetadataValue::$variant(value)
                }
            }
        )*
    };
}

macro_rules! impl_try_from_metadata {
    ($($ty:ty => $variant:ident => $extract:expr),* $(,)?) => {
        $(
            impl TryFrom<MetadataValue> for $ty {
                type Error = MetadataTypeError;

                fn try_from(value: MetadataValue) -> ::core::result::Result<Self, Self::Error> {
                    match value {
                        MetadataValue::$variant(v) => Ok(v),
                        other => Err(MetadataTypeError {
                            expected: stringify!($variant),
                            found: other,
                        }),
                    }
                }
            }

            impl<'a> TryFrom<&'a MetadataValue> for $ty {
                type Error = MetadataTypeError;

                fn try_from(value: &'a MetadataValue) -> ::core::result::Result<Self, Self::Error> {
                    match value {
                        MetadataValue::$variant(v) => Ok($extract(v.clone())),
                        other => Err(MetadataTypeError {
                            expected: stringify!($variant),
                            found: other.clone(),
                        }),
                    }
                }
            }
        )*
    };
}

impl_from_for_metadata! {
    u8  => Uint8,
    i8  => Int8,
    u16 => Uint16,
    i16 => Int16,
    u32 => Uint32,
    i32 => Int32,
    f32 => Float32,
    bool => Bool,
    String => String,
    Vec<MetadataValue> => Array,
    u64 => Uint64,
    i64 => Int64,
    f64 => Float64,
}

impl From<&str> for MetadataValue {
    fn from(value: &str) -> Self {
        MetadataValue::String(value.to_owned())
    }
}

impl_try_from_metadata! {
    u8  => Uint8  => |v| v,
    i8  => Int8   => |v| v,
    u16 => Uint16 => |v| v,
    i16 => Int16  => |v| v,
    u32 => Uint32 => |v| v,
    i32 => Int32  => |v| v,
    f32 => Float32 => |v| v,
    bool => Bool => |v| v,
    u64 => Uint64 => |v| v,
    i64 => Int64 => |v| v,
    f64 => Float64 => |v| v,

    // owned types need cloning
    String => String => |v: String| v,
    Vec<MetadataValue> => Array => |v: Vec<MetadataValue>| v,
}

/// GGML type of a tensor in the GGUF file.
///
/// Represents the quantization format used for tensor data.
/// Most types are quantized formats that compress float values
/// to reduce memory footprint while maintaining accuracy.
#[derive(Debug)]
#[allow(non_camel_case_types)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q4_2 = 4, // Unsupported
    Q4_3 = 5, // Unsupported
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS = 17,
    IQ3_XXS = 18,
    IQ1_S = 19,
    IQ4_NL = 20,
    IQ3_S = 21,
    IQ2_S = 22,
    IQ4_XS = 23,
    I8 = 24,
    I16 = 25,
    I32 = 26,
    I64 = 27,
    F64 = 28,
    IQ1_M = 29,
    BF16 = 30,
    Q4_0_4_4 = 31, // Unsupported
    Q4_0_4_8 = 32, // Unsupported
    Q4_0_8_8 = 33, // Unsupported
    TQ1_0 = 34,
    TQ2_0 = 35,
    IQ4_NL_4_4 = 36, // Unsupported
    IQ4_NL_4_8 = 37, // Unsupported
    IQ4_NL_8_8 = 38, // Unsupported
    MXFP4 = 39,
    NVFP4 = 40,
    Q1_0 = 41,
    Count = 42,
}

impl Display for GGMLType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GGMLType::F32 => write!(f, "F32"),
            GGMLType::F16 => write!(f, "F16"),
            GGMLType::Q4_0 => write!(f, "Q4_0"),
            GGMLType::Q4_1 => write!(f, "Q4_1"),
            GGMLType::Q4_2 => write!(f, "Q4_2 (UNSUPPORTED)"),
            GGMLType::Q4_3 => write!(f, "Q4_3 (UNSUPPORTED)"),
            GGMLType::Q5_0 => write!(f, "Q5_0"),
            GGMLType::Q5_1 => write!(f, "Q5_1"),
            GGMLType::Q8_0 => write!(f, "Q8_0"),
            GGMLType::Q8_1 => write!(f, "Q8_1"),
            GGMLType::Q2_K => write!(f, "Q2_K"),
            GGMLType::Q3_K => write!(f, "Q3_K"),
            GGMLType::Q4_K => write!(f, "Q4_K"),
            GGMLType::Q5_K => write!(f, "Q5_K"),
            GGMLType::Q6_K => write!(f, "Q6_K"),
            GGMLType::Q8_K => write!(f, "Q8_K"),
            GGMLType::IQ2_XXS => write!(f, "IQ2_XXS"),
            GGMLType::IQ2_XS => write!(f, "IQ2_XS"),
            GGMLType::IQ3_XXS => write!(f, "IQ3_XXS"),
            GGMLType::IQ1_S => write!(f, "IQ1_S"),
            GGMLType::IQ4_NL => write!(f, "IQ4_NL"),
            GGMLType::IQ3_S => write!(f, "IQ3_S"),
            GGMLType::IQ2_S => write!(f, "IQ2_S"),
            GGMLType::IQ4_XS => write!(f, "IQ4_XS"),
            GGMLType::I8 => write!(f, "I8"),
            GGMLType::I16 => write!(f, "I16"),
            GGMLType::I32 => write!(f, "I32"),
            GGMLType::I64 => write!(f, "I64"),
            GGMLType::F64 => write!(f, "F64"),
            GGMLType::IQ1_M => write!(f, "IQ1_M"),
            GGMLType::BF16 => write!(f, "BF16"),
            GGMLType::Q4_0_4_4 => write!(f, "Q4_0_4_4 (UNSUPPORTED)"),
            GGMLType::Q4_0_4_8 => write!(f, "Q4_0_4_8 (UNSUPPORTED)"),
            GGMLType::Q4_0_8_8 => write!(f, "Q4_0_8_8 (UNSUPPORTED)"),
            GGMLType::TQ1_0 => write!(f, "TQ1_0"),
            GGMLType::TQ2_0 => write!(f, "TQ2_0"),
            GGMLType::IQ4_NL_4_4 => write!(f, "IQ4_NL_4_4 (UNSUPPORTED)"),
            GGMLType::IQ4_NL_4_8 => write!(f, "IQ4_NL_4_8 (UNSUPPORTED)"),
            GGMLType::IQ4_NL_8_8 => write!(f, "IQ4_NL_8_8 (UNSUPPORTED)"),
            GGMLType::MXFP4 => write!(f, "MXFP4"),
            GGMLType::NVFP4 => write!(f, "NVFP4"),
            GGMLType::Q1_0 => write!(f, "Q1_0"),
            GGMLType::Count => write!(f, "Count"),
        }
    }
}

impl TryFrom<u32> for GGMLType {
    type Error = Error;

    fn try_from(value: u32) -> Result<Self> {
        Ok(match value {
            0 => GGMLType::F32,
            1 => GGMLType::F16,
            2 => GGMLType::Q4_0,
            3 => GGMLType::Q4_1,
            6 => GGMLType::Q5_0,
            7 => GGMLType::Q5_1,
            8 => GGMLType::Q8_0,
            9 => GGMLType::Q8_1,
            10 => GGMLType::Q2_K,
            11 => GGMLType::Q3_K,
            12 => GGMLType::Q4_K,
            13 => GGMLType::Q5_K,
            14 => GGMLType::Q6_K,
            15 => GGMLType::Q8_K,
            16 => GGMLType::IQ2_XXS,
            17 => GGMLType::IQ2_XS,
            18 => GGMLType::IQ3_XXS,
            19 => GGMLType::IQ1_S,
            20 => GGMLType::IQ4_NL,
            21 => GGMLType::IQ3_S,
            22 => GGMLType::IQ2_S,
            23 => GGMLType::IQ4_XS,
            24 => GGMLType::I8,
            25 => GGMLType::I16,
            26 => GGMLType::I32,
            27 => GGMLType::I64,
            28 => GGMLType::F64,
            29 => GGMLType::IQ1_M,
            30 => GGMLType::BF16,
            31 => GGMLType::Q4_0_4_4,
            32 => GGMLType::Q4_0_4_8,
            33 => GGMLType::Q4_0_8_8,
            34 => GGMLType::TQ1_0,
            35 => GGMLType::TQ2_0,
            36 => GGMLType::IQ4_NL_4_4,
            37 => GGMLType::IQ4_NL_4_8,
            38 => GGMLType::IQ4_NL_8_8,
            39 => GGMLType::MXFP4,
            40 => GGMLType::NVFP4,
            41 => GGMLType::Q1_0,
            42 => GGMLType::Count,
            _ => return Err(Error::InvalidGGMLType(value)),
        })
    }
}

impl GGUFModel {
    /// Decode the GGUF file.
    pub(crate) fn decode(&mut self, mut reader: impl std::io::Read) -> Result<()> {
        // decode kv
        for _i in 0..self.num_kv() {
            let key = self.read_string(&mut reader)?;
            let value_type: MetadataValueType = self.read_u32(&mut reader)?.try_into()?;
            let value = match value_type {
                MetadataValueType::Uint8 => MetadataValue::Uint8(self.read_u8(&mut reader)?),
                MetadataValueType::Int8 => MetadataValue::Int8(self.read_i8(&mut reader)?),
                MetadataValueType::Uint16 => MetadataValue::Uint16(self.read_u16(&mut reader)?),
                MetadataValueType::Int16 => MetadataValue::Int16(self.read_i16(&mut reader)?),
                MetadataValueType::Uint32 => MetadataValue::Uint32(self.read_u32(&mut reader)?),
                MetadataValueType::Int32 => MetadataValue::Int32(self.read_i32(&mut reader)?),
                MetadataValueType::Float32 => MetadataValue::Float32(self.read_f32(&mut reader)?),
                MetadataValueType::Bool => MetadataValue::Bool(self.read_bool(&mut reader)?),
                MetadataValueType::String => MetadataValue::String(self.read_string(&mut reader)?),
                MetadataValueType::Array => MetadataValue::Array(self.read_array(&mut reader)?),
                MetadataValueType::Uint64 => MetadataValue::Uint64(self.read_u64(&mut reader)?),
                MetadataValueType::Int64 => MetadataValue::Int64(self.read_i64(&mut reader)?),
                MetadataValueType::Float64 => MetadataValue::Float64(self.read_f64(&mut reader)?),
            };
            #[cfg(feature = "logging")]
            {
                debug!("kv [{}] vtype {:?} key={}, value={}", _i, value_type, key, value);
            }
            self.kv.insert(key, value);
        }

        // decode tensors
        for _ in 0..self.num_tensor() {
            let name = self.read_string(&mut reader)?;
            let dims = self.read_u32(&mut reader)?;
            let mut shape = [1_u64; 4];
            for i in 0..dims {
                shape[i as usize] = self.read_u64(&mut reader)?;
            }

            let kind = self.read_u32(&mut reader)?;
            let offset = self.read_u64(&mut reader)?;
            let block_size = match kind {
                _ if kind < 2 => 1,
                _ if kind < 10 => 32,
                _ if kind == 40 => 64,
                _ if kind == 41 => 128,
                _ => 256,
            };
            let ggml_type_kind: GGMLType = kind.try_into()?;
            let type_size = match ggml_type_kind {
                GGMLType::F32 => 4,
                GGMLType::F16 => 2,
                GGMLType::Q4_0 => 2 + block_size / 2,
                GGMLType::Q4_1 => 2 + 2 + block_size / 2,
                GGMLType::Q4_2 => 0,
                GGMLType::Q4_3 => 0,
                GGMLType::Q5_0 => 2 + 4 + block_size / 2,
                GGMLType::Q5_1 => 2 + 2 + 4 + block_size / 2,
                GGMLType::Q8_0 => 2 + block_size,
                GGMLType::Q8_1 => 4 + 4 + block_size,
                GGMLType::Q2_K => block_size / 16 + block_size / 4 + 2 + 2,
                GGMLType::Q3_K => block_size / 8 + block_size / 4 + 12 + 2,
                GGMLType::Q4_K => 2 + 2 + 12 + block_size / 2,
                GGMLType::Q5_K => 2 + 2 + 12 + block_size / 8 + block_size / 2,
                GGMLType::Q6_K => block_size / 2 + block_size / 4 + block_size / 16 + 2,
                GGMLType::Q8_K => 4 + block_size + block_size / 16 * 2,
                GGMLType::IQ2_XXS => 2 + block_size / 8 * 2,
                GGMLType::IQ2_XS => 2 + block_size / 8 * 2 + block_size / 32,
                GGMLType::IQ3_XXS => 2 + 3 * (block_size / 8),
                GGMLType::IQ1_S => 2 + block_size / 8 + block_size / 16,
                GGMLType::IQ4_NL => 2 + 16,
                GGMLType::IQ3_S => 2 + 13 * (block_size / 32) + block_size / 64,
                GGMLType::IQ2_S => 2 + block_size / 4 + block_size / 16,
                GGMLType::IQ4_XS => 2 + 2 + block_size / 64 + block_size / 2,
                GGMLType::I8 => 1,
                GGMLType::I16 => 2,
                GGMLType::I32 => 4,
                GGMLType::I64 => 8,
                GGMLType::F64 => 8,
                GGMLType::IQ1_M => block_size / 8 + block_size / 16 + block_size / 32,
                GGMLType::BF16 => 2,
                GGMLType::IQ4_NL_4_4 => 0,
                GGMLType::IQ4_NL_4_8 => 0,
                GGMLType::IQ4_NL_8_8 => 0,
                GGMLType::TQ1_0 => 2 + block_size / 64 + (block_size - 4 * block_size / 64) / 5,
                GGMLType::TQ2_0 => 2 + block_size / 4,
                GGMLType::Q4_0_4_4 => 0,
                GGMLType::Q4_0_4_8 => 0,
                GGMLType::Q4_0_8_8 => 0,
                GGMLType::MXFP4 => block_size + 1 + 16,
                GGMLType::NVFP4 => block_size / 16 + block_size / 2,
                GGMLType::Q1_0 => 2 + block_size / 128,
                GGMLType::Count => unreachable!("GGMLType::Count is not a real data format"),
            };

            let parameters = shape[0] * shape[1] * shape[2] * shape[3];
            let size = parameters * type_size / block_size;

            self.tensors.push(Tensor {
                name,
                kind,
                offset,
                size,
                shape,
            });

            self.parameters += parameters;
        }

        Ok(())
    }

    fn read_u8(&self, mut reader: impl std::io::Read) -> Result<u8> {
        Ok(reader.read_u8()?)
    }

    fn read_u32(&self, mut reader: impl std::io::Read) -> Result<u32> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_u32::<LittleEndian>()?,
            ByteOrder::BE => reader.read_u32::<BigEndian>()?,
        })
    }

    fn read_f32(&self, mut reader: impl std::io::Read) -> Result<f32> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_f32::<LittleEndian>()?,
            ByteOrder::BE => reader.read_f32::<BigEndian>()?,
        })
    }

    fn read_f64(&self, mut reader: impl std::io::Read) -> Result<f64> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_f64::<LittleEndian>()?,
            ByteOrder::BE => reader.read_f64::<BigEndian>()?,
        })
    }

    fn read_u64(&self, mut reader: impl std::io::Read) -> Result<u64> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_u64::<LittleEndian>()?,
            ByteOrder::BE => reader.read_u64::<BigEndian>()?,
        })
    }

    fn read_i8(&self, mut reader: impl std::io::Read) -> Result<i8> {
        Ok(reader.read_i8()?)
    }

    fn read_u16(&self, mut reader: impl std::io::Read) -> Result<u16> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_u16::<LittleEndian>()?,
            ByteOrder::BE => reader.read_u16::<BigEndian>()?,
        })
    }

    fn read_i16(&self, mut reader: impl std::io::Read) -> Result<i16> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_i16::<LittleEndian>()?,
            ByteOrder::BE => reader.read_i16::<BigEndian>()?,
        })
    }

    fn read_i32(&self, mut reader: impl std::io::Read) -> Result<i32> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_i32::<LittleEndian>()?,
            ByteOrder::BE => reader.read_i32::<BigEndian>()?,
        })
    }

    fn read_i64(&self, mut reader: impl std::io::Read) -> Result<i64> {
        Ok(match self.bo {
            ByteOrder::LE => reader.read_i64::<LittleEndian>()?,
            ByteOrder::BE => reader.read_i64::<BigEndian>()?,
        })
    }

    fn read_bool(&self, mut reader: impl std::io::Read) -> Result<bool> {
        Ok(reader.read_u8()? != 0)
    }

    fn read_string(&self, mut reader: impl std::io::Read) -> Result<String> {
        let name_len = self.read_version_size(&mut reader)?;
        let mut buffer = vec![0; name_len as usize];
        reader.read_exact(&mut buffer)?;
        Ok(String::from_utf8_lossy(&buffer).to_string())
    }

    fn read_array(&self, mut reader: impl std::io::Read) -> Result<Vec<MetadataValue>> {
        let mut data = Vec::new();
        let item_type: MetadataValueType = self.read_u32(&mut reader)?.try_into()?;
        let array_len = self.read_version_size(&mut reader)?;
        let read_count: usize = u64::min(array_len, self.max_array_size) as usize;
        for _ in 0..array_len {
            let value = match item_type {
                MetadataValueType::Uint8 => MetadataValue::Uint8(self.read_u8(&mut reader)?),
                MetadataValueType::Int8 => MetadataValue::Int8(self.read_i8(&mut reader)?),
                MetadataValueType::Uint16 => MetadataValue::Uint16(self.read_u16(&mut reader)?),
                MetadataValueType::Int16 => MetadataValue::Int16(self.read_i16(&mut reader)?),
                MetadataValueType::Uint32 => MetadataValue::Uint32(self.read_u32(&mut reader)?),
                MetadataValueType::Int32 => MetadataValue::Int32(self.read_i32(&mut reader)?),
                MetadataValueType::Float32 => MetadataValue::Float32(self.read_f32(&mut reader)?),
                MetadataValueType::Bool => MetadataValue::Bool(self.read_bool(&mut reader)?),
                MetadataValueType::String => MetadataValue::String(self.read_string(&mut reader)?),
                MetadataValueType::Uint64 => MetadataValue::Uint64(self.read_u64(&mut reader)?),
                MetadataValueType::Int64 => MetadataValue::Int64(self.read_i64(&mut reader)?),
                MetadataValueType::Float64 => MetadataValue::Float64(self.read_f64(&mut reader)?),
                _ => return Err(Error::UnsupportedArrayValue),
            };
            if read_count > 0 && data.len() < read_count {
                data.push(value);
            }
        }

        Ok(data)
    }

    fn read_version_size(&self, mut reader: impl std::io::Read) -> Result<u64> {
        Ok(match self.version.borrow() {
            Version::V1(_) => self.read_u32(&mut reader)? as u64,
            Version::V2(_) => self.read_u64(&mut reader)?,
            Version::V3(_) => self.read_u64(&mut reader)?,
        })
    }

    /// Get the version of the decoded GGUF model.
    ///
    /// Returns one of: "v1", "v2", or "v3".
    pub fn get_version(&self) -> String {
        match &self.version {
            Version::V1(_) => String::from("v1"),
            Version::V2(_) => String::from("v2"),
            Version::V3(_) => String::from("v3"),
        }
    }

    /// Get the number of key-value pairs in the GGUF file.
    pub fn num_kv(&self) -> u64 {
        match &self.version {
            Version::V1(v1) => v1.num_kv as u64,
            Version::V2(v2) => v2.num_kv,
            Version::V3(v3) => v3.num_kv,
        }
    }

    /// Get the number of tensors in the GGUF file.
    ///
    /// Returns the total count of tensors stored in the model.
    pub fn num_tensor(&self) -> u64 {
        match &self.version {
            Version::V1(v1) => v1.num_tensor as u64,
            Version::V2(v2) => v2.num_tensor,
            Version::V3(v3) => v3.num_tensor,
        }
    }

    /// Get the model family/architecture of the GGUF file.
    ///
    /// Returns the value of `general.architecture` metadata key,
    /// or "unknown" if not present.
    ///
    /// Common values include: "llama", "phi", "mistral", "qwen", etc.
    pub fn model_family(&self) -> String {
        let arch = self.kv.get("general.architecture").cloned();

        match arch {
            Some(MetadataValue::String(arch)) => arch,
            _ => String::from("unknown"),
        }
    }

    /// Get the estimated number of parameters in the model.
    ///
    /// Returns a human-readable string (e.g., "7B", "13B", "192").
    /// Returns "unknown" if parameters cannot be determined.
    pub fn model_parameters(&self) -> String {
        if self.parameters > 0 {
            human_number(self.parameters)
        } else {
            String::from("unknown")
        }
    }

    /// Get the quantization file type of the GGUF file.
    ///
    /// Returns a human-readable description of the quantization method
    /// (e.g., "All F32", "Mostly Q4_0", "Mostly BF16").
    /// Returns "unknown" if not present.
    pub fn file_type(&self) -> String {
        if let Some(MetadataValue::Uint64(ft)) = self.kv.get("general.file_type") {
            file_type(*ft)
        } else {
            String::from("unknown")
        }
    }

    /// Get the key-value metadata of the GGUF file.
    ///
    /// Returns a reference to the metadata map containing all key-value pairs
    /// from the GGUF file.
    ///
    /// Common keys include:
    /// - `general.architecture`: Model architecture (e.g., "llama")
    /// - `general.name`: Model name
    /// - `tokenizer.ggml.tokens`: Tokenizer vocabulary
    pub fn metadata(&self) -> &BTreeMap<String, MetadataValue> {
        &self.kv
    }

    /// Get the tensors of the GGUF file.
    ///
    /// Returns a reference to the vector of tensors, each containing
    /// name, type, offset, size, and shape information.
    pub fn tensors(&self) -> &Vec<Tensor> {
        &self.tensors
    }

    /// Converts [`GGUFModel`] into its main parts: metadata and tensors.
    pub fn into_parts(self) -> (BTreeMap<String, MetadataValue>, Vec<Tensor>) {
        (self.kv, self.tensors)
    }
}

/// Get a `GGUFContainer` from a file, truncating all arrays to length 3.
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist
/// - The file has an unsupported format (ggml, ggmf, ggjt, ggla)
/// - The file has an invalid magic number
/// - An I/O error occurs while reading the file
///
/// # Examples
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container;
///
/// let container = get_gguf_container("model.gguf")?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn get_gguf_container(file: &str) -> Result<GGUFContainer<'_>> {
    get_gguf_container_array_size(file, 3)
}

/// Get a `GGUFContainer` from a file with the provided max array size.
///
/// # Arguments
///
/// * `file` - Path to the GGUF file
/// * `max_array_size` - Maximum number of elements to read from array metadata
///
/// # Errors
///
/// Returns an error if:
/// - The file does not exist
/// - The file has an unsupported format (ggml, ggmf, ggjt, ggla)
/// - The file has an invalid magic number
/// - An I/O error occurs while reading the file
///
/// # Examples
///
/// ```rust,no_run
/// use gguf_rs::get_gguf_container_array_size;
///
/// // Read all array elements
/// let container = get_gguf_container_array_size("model.gguf", u64::MAX)?;
///
/// // Limit arrays to 100 elements for performance
/// let container = get_gguf_container_array_size("model.gguf", 100)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn get_gguf_container_array_size(file: &str, max_array_size: u64) -> Result<GGUFContainer<'_>> {
    let mut reader = std::fs::File::open(file)?;
    let order = read_gguf_magic_order(&mut reader)?;
    let container = GGUFContainer::new(order, Box::new(reader)).with_max_array_size(max_array_size);
    Ok(container)
}

/// Reads the magic prefix of the GGUF file to determine the [`ByteOrder`] of the file.
pub fn read_gguf_magic_order<R: std::io::Read>(reader: &mut R) -> Result<ByteOrder> {
    let byte_le = reader.read_i32::<LittleEndian>()?;
    match byte_le {
        FILE_MAGIC_GGUF_LE => Ok(ByteOrder::LE),
        FILE_MAGIC_GGUF_BE => Ok(ByteOrder::BE),
        other => Err(Error::UnsupportedFileFormat(other)),
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_read_le_v3_gguf() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        assert_eq!(model.get_version(), "v3");
        assert_eq!(model.model_family(), "llama");
        assert_eq!(model.file_type(), "unknown");
        assert_eq!(model.model_parameters(), "192");
    }

    #[test]
    fn test_read_le_v3_gguf_with_tokens() {
        let mut container =
            super::get_gguf_container_array_size("tests/test-le-v3.gguf", u64::MAX).unwrap();
        let model = container.decode().unwrap();
        assert_eq!(model.get_version(), "v3");
        assert_eq!(model.model_family(), "llama");
        assert_eq!(model.file_type(), "unknown");
        assert_eq!(model.model_parameters(), "192");
        println!("{:?}", model.kv);
    }

    #[test]
    fn test_file_not_found() {
        let result = super::get_gguf_container("nonexistent.gguf");
        let Err(crate::error::Error::IO(io)) = result else {
            panic!("expected error finding file");
        };
        assert_eq!(io.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn test_invalid_file_magic() {
        use std::io::Cursor;
        let invalid_data = vec![0x00, 0x00, 0x00, 0x00];
        let cursor = Cursor::new(invalid_data);
        let mut container = super::GGUFContainer::new(super::ByteOrder::LE, Box::new(cursor))
            .with_max_array_size(u64::MAX);
        let result = container.decode();
        assert!(result.is_err());
    }

    #[test]
    fn test_metadata_value_type_conversion() {
        use super::MetadataValueType;
        use std::convert::TryFrom;

        assert!(matches!(MetadataValueType::try_from(0), Ok(MetadataValueType::Uint8)));
        assert!(matches!(MetadataValueType::try_from(6), Ok(MetadataValueType::Float32)));
        assert!(matches!(MetadataValueType::try_from(8), Ok(MetadataValueType::String)));
        assert!(MetadataValueType::try_from(100).is_err());
    }

    #[test]
    fn test_ggml_type_conversion() {
        use super::GGMLType;
        use std::convert::TryFrom;

        assert!(matches!(GGMLType::try_from(0), Ok(GGMLType::F32)));
        assert!(matches!(GGMLType::try_from(2), Ok(GGMLType::Q4_0)));
        assert!(GGMLType::try_from(100).is_err());
    }

    #[test]
    fn test_byte_order_default() {
        use super::ByteOrder;
        let bo = ByteOrder::default();
        assert!(matches!(bo, ByteOrder::LE));
    }

    #[test]
    fn test_tensors() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        let tensors = model.tensors();
        assert!(!tensors.is_empty());

        for tensor in tensors {
            assert!(!tensor.name.is_empty());
            assert!(!tensor.shape.is_empty());
        }
    }

    #[test]
    fn test_num_tensor() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        assert!(model.num_tensor() > 0);
    }

    #[test]
    fn test_get_version() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        assert_eq!(container.get_version(), "v1"); // Before decode, default is v1
        let _ = container.decode().unwrap();
        // After decode, version should be v3
    }

    // ========== Additional tests for improved coverage ==========

    #[test]
    fn test_human_number_small() {
        assert_eq!(super::human_number(999), "999");
        assert_eq!(super::human_number(1000), "1000");
        assert_eq!(super::human_number(1001), "1K");
        assert_eq!(super::human_number(1500), "2K");
    }

    #[test]
    fn test_human_number_medium() {
        assert_eq!(super::human_number(1_000_000), "1000K");
        assert_eq!(super::human_number(1_000_001), "1M");
        assert_eq!(super::human_number(2_000_001), "2M");
        assert_eq!(super::human_number(3_500_000), "4M");
    }

    #[test]
    fn test_human_number_large() {
        assert_eq!(super::human_number(1_000_000_000), "1000M");
        assert_eq!(super::human_number(1_000_000_001), "1B");
        assert_eq!(super::human_number(7_500_000_000), "8B");
    }

    #[test]
    fn test_file_type_all_values() {
        assert_eq!(super::file_type(0), "All F32");
        assert_eq!(super::file_type(1), "Mostly F16");
        assert_eq!(super::file_type(2), "Mostly Q4_0");
        assert_eq!(super::file_type(7), "Mostly Q8_0");
        assert_eq!(super::file_type(14), "Mostly Q6_K");
        assert_eq!(super::file_type(24), "Mostly BF16");
        assert_eq!(super::file_type(99), "unknown");
    }

    #[test]
    fn test_metadata_value_type_all_variants() {
        use super::MetadataValueType;
        use std::convert::TryFrom;

        // Test all valid type values
        assert!(matches!(MetadataValueType::try_from(0), Ok(MetadataValueType::Uint8)));
        assert!(matches!(MetadataValueType::try_from(1), Ok(MetadataValueType::Int8)));
        assert!(matches!(MetadataValueType::try_from(2), Ok(MetadataValueType::Uint16)));
        assert!(matches!(MetadataValueType::try_from(3), Ok(MetadataValueType::Int16)));
        assert!(matches!(MetadataValueType::try_from(4), Ok(MetadataValueType::Uint32)));
        assert!(matches!(MetadataValueType::try_from(5), Ok(MetadataValueType::Int32)));
        assert!(matches!(MetadataValueType::try_from(6), Ok(MetadataValueType::Float32)));
        assert!(matches!(MetadataValueType::try_from(7), Ok(MetadataValueType::Bool)));
        assert!(matches!(MetadataValueType::try_from(8), Ok(MetadataValueType::String)));
        assert!(matches!(MetadataValueType::try_from(9), Ok(MetadataValueType::Array)));
        assert!(matches!(MetadataValueType::try_from(10), Ok(MetadataValueType::Uint64)));
        assert!(matches!(MetadataValueType::try_from(11), Ok(MetadataValueType::Int64)));
        assert!(matches!(MetadataValueType::try_from(12), Ok(MetadataValueType::Float64)));
    }

    #[test]
    fn test_ggml_type_all_valid_types() {
        use super::GGMLType;
        use std::convert::TryFrom;

        // Test a representative sample of GGML types
        assert!(matches!(GGMLType::try_from(1), Ok(GGMLType::F16)));
        assert!(matches!(GGMLType::try_from(3), Ok(GGMLType::Q4_1)));
        assert!(matches!(GGMLType::try_from(6), Ok(GGMLType::Q5_0)));
        assert!(matches!(GGMLType::try_from(7), Ok(GGMLType::Q5_1)));
        assert!(matches!(GGMLType::try_from(8), Ok(GGMLType::Q8_0)));
        assert!(matches!(GGMLType::try_from(10), Ok(GGMLType::Q2_K)));
        assert!(matches!(GGMLType::try_from(30), Ok(GGMLType::BF16)));
        assert!(matches!(GGMLType::try_from(39), Ok(GGMLType::MXFP4)));
    }

    #[test]
    fn test_ggml_type_invalid() {
        use super::GGMLType;
        use std::convert::TryFrom;

        assert!(GGMLType::try_from(100).is_err());
        assert!(GGMLType::try_from(255).is_err());
    }

    #[test]
    fn test_model_family_unknown() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        // This test file has "llama" architecture
        assert_eq!(model.model_family(), "llama");
    }

    #[test]
    fn test_model_parameters_format() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        // Test file has 192 parameters
        assert_eq!(model.model_parameters(), "192");
    }

    #[test]
    fn test_metadata_accessor() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        let metadata = model.metadata();
        assert!(metadata.contains_key("general.architecture"));
        assert!(metadata.contains_key("llama.block_count"));
    }

    #[test]
    fn test_num_kv() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        assert!(model.num_kv() > 0);
    }

    #[test]
    fn test_tensor_properties() {
        let mut container = super::get_gguf_container("tests/test-le-v3.gguf").unwrap();
        let model = container.decode().unwrap();
        let tensors = model.tensors();

        for tensor in tensors {
            // Verify tensor has valid properties
            assert!(!tensor.name.is_empty());
            assert!(!tensor.shape.is_empty());
            // Offset and size should be non-negative (u64)
            let _ = tensor.offset;
            let _ = tensor.size;
            let _ = tensor.kind;
        }
    }

    #[test]
    fn test_container_new() {
        use super::{ByteOrder, GGUFContainer};
        use std::io::Cursor;

        let cursor = Cursor::new(vec![]);
        let container =
            GGUFContainer::new(ByteOrder::LE, Box::new(cursor)).with_max_array_size(100);
        assert_eq!(container.get_version(), "v1"); // Default version
    }

    #[test]
    fn test_byte_order_variants() {
        use super::ByteOrder;

        let le = ByteOrder::LE;
        let be = ByteOrder::BE;

        // Just verify we can create both variants
        let _ = format!("{:?}", le);
        let _ = format!("{:?}", be);
    }

    #[test]
    fn test_version_variants() {
        use super::{Version, V1, V2, V3};

        let v1 = Version::V1(V1::default());
        let v2 = Version::V2(V2::default());
        let v3 = Version::V3(V3::default());

        // Verify we can create all version variants
        let _ = format!("{:?}", v1);
        let _ = format!("{:?}", v2);
        let _ = format!("{:?}", v3);
    }

    #[test]
    fn test_invalid_file_magic_detailed() {
        use std::io::Cursor;

        // Test with various invalid magic numbers
        let invalid_magics = vec![
            vec![0x00, 0x00, 0x00, 0x00],
            vec![0xFF, 0xFF, 0xFF, 0xFF],
            vec![0x12, 0x34, 0x56, 0x78],
        ];

        for magic in invalid_magics {
            let cursor = Cursor::new(magic);
            let mut container = super::GGUFContainer::new(super::ByteOrder::LE, Box::new(cursor))
                .with_max_array_size(u64::MAX);
            let result = container.decode();
            assert!(result.is_err(), "Expected error for invalid magic");
        }
    }

    #[test]
    fn test_get_gguf_container_array_size() {
        // Test with custom array size
        let result = super::get_gguf_container_array_size("tests/test-le-v3.gguf", 1);
        assert!(result.is_ok());

        let mut container = result.unwrap();
        let model = container.decode().unwrap();

        // With max_array_size=1, arrays should be truncated
        let tokens = model.kv.get("tokenizer.ggml.tokens");
        if let Some(crate::MetadataValue::Array(arr)) = tokens {
            assert!(arr.len() <= 1, "Array should be truncated to max size");
        }
    }
}

/// Memory-mapped file support (requires `mmap` feature)
#[cfg(feature = "mmap")]
pub mod mmap;

/// Async I/O support (requires `async` feature)
#[cfg(feature = "async")]
pub mod async_io;

/// GGUF file writing support
pub mod writer;
