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
