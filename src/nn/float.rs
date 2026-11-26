//! Abstracción mínima sobre `f32`/`f64`: se entrena en `f32` (más rápido,
//! vectoriza mejor) y se comprueban gradientes en `f64` (más precisión).

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Número de punto flotante usable por las redes.
pub trait Float:
    Copy
    + Default
    + Debug
    + PartialOrd
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
{
    /// Cero.
    const ZERO: Self;
    /// Uno.
    const ONE: Self;
    /// Conversión desde `f64`.
    #[must_use]
    fn from_f64(x: f64) -> Self;
    /// Conversión a `f64`.
    fn to_f64(self) -> f64;
    /// Exponencial.
    #[must_use]
    fn exp(self) -> Self;
    /// Logaritmo natural.
    #[must_use]
    fn ln(self) -> Self;
    /// Tangente hiperbólica.
    #[must_use]
    fn tanh(self) -> Self;
    /// Raíz cuadrada.
    #[must_use]
    fn sqrt(self) -> Self;
    /// Máximo.
    #[must_use]
    fn max(self, other: Self) -> Self;
}

macro_rules! impl_float {
    ($t:ty) => {
        impl Float for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            #[inline]
            fn from_f64(x: f64) -> Self {
                x as $t
            }
            #[inline]
            fn to_f64(self) -> f64 {
                f64::from(self)
            }
            #[inline]
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            #[inline]
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            #[inline]
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
            #[inline]
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            #[inline]
            fn max(self, other: Self) -> Self {
                <$t>::max(self, other)
            }
        }
    };
}

impl_float!(f32);
impl_float!(f64);
