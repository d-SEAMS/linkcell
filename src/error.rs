//! Recoverable failures from a neighbour search.

use std::fmt;

/// Why a search could not run.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Error {
    /// `k == 0`.
    ZeroK,
    /// A box length is not strictly positive, or H is singular.
    BadBox,
    /// `n == 0` or the point list is empty. Not a buffer-size mismatch.
    Empty,
    /// Caller `out` length is not `n * k`.
    BufferSize,
    /// `n * k` does not fit a slice (`usize` or `isize`).
    Overflow,
    /// `mask` is `Some` and its length is not `n`.
    MaskLen,
    /// Linked-cell mesh would overflow or exceed the bin cap.
    TooManyCells,
    /// `cutoff` is not finite and strictly positive.
    BadCutoff,
    /// Periodic image stencil exceeds the 2^24 resource cap.
    TooManyImages,
    /// An axis is not periodic and `libvesin.so` is not loaded.
    Partial,
    /// The device is not the CPU and `libvesin.so` is not loaded.
    Device,
    /// A point has more neighbours than the requested pad.
    Padded,
    /// A coordinate is not finite.
    NonFinite,
    /// The part count is zero, or the balance bound is below one.
    BadPartition,
    /// The cell graph is not a square CSR of that grid.
    BadGraph,
    /// A device curve split was requested and no device is loaded.
    NoDevice,
    /// The device curve split disagreed with the host weights.
    DeviceMismatch,
    /// Scotch support was not compiled in.
    ScotchDisabled,
    /// The Scotch library rejected the graph.
    ScotchFailed,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::ZeroK => write!(f, "k must be at least 1"),
            Error::BadBox => write!(f, "singular or non-positive cell"),
            Error::Empty => write!(f, "no points"),
            Error::BufferSize => write!(f, "out buffer length must be n * k"),
            Error::Overflow => write!(f, "n * k overflows"),
            Error::MaskLen => write!(f, "mask length must be n"),
            Error::TooManyCells => write!(f, "linked-cell mesh is too fine"),
            Error::BadCutoff => write!(f, "cutoff must be finite and positive"),
            Error::TooManyImages => write!(f, "periodic image count exceeds 2^24"),
            Error::Partial => write!(
                f,
                "partial periodicity needs libvesin.so on the loader path"
            ),
            Error::Device => write!(f, "this device needs libvesin.so on the loader path"),
            Error::Padded => write!(f, "a point has more neighbours than padded_to"),
            Error::NonFinite => write!(f, "a coordinate is not finite"),
            Error::BadPartition => {
                write!(f, "part count must be positive and balance at least 1")
            }
            Error::BadGraph => write!(f, "cell graph is not a square CSR"),
            Error::NoDevice => write!(f, "CUDA curve split is not available"),
            Error::DeviceMismatch => {
                write!(f, "device curve split disagreed with the host weights")
            }
            Error::ScotchDisabled => write!(f, "Scotch support is compiled out"),
            Error::ScotchFailed => write!(f, "Scotch returned an error"),
        }
    }
}

impl std::error::Error for Error {}

impl From<minimage::Error> for Error {
    fn from(err: minimage::Error) -> Self {
        match err {
            minimage::Error::BadBox => Error::BadBox,
            minimage::Error::BufferSize => Error::BufferSize,
            minimage::Error::Empty => Error::Empty,
        }
    }
}
