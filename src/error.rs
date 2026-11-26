//! Tipo de error del crate.

use std::fmt;
use std::io;

/// Errores recuperables: E/S, formatos de archivo y configuración.
///
/// Los estados imposibles del núcleo numérico se tratan con `debug_assert!`
/// y pánicos documentados, no con este tipo.
#[derive(Debug)]
pub enum Error {
    /// Error de entrada/salida del sistema operativo.
    Io(io::Error),
    /// Archivo con formato inválido (PNG, DEFLATE, nivel en texto, ...).
    Format {
        /// Qué formato se estaba leyendo.
        what: &'static str,
        /// Descripción del problema.
        detail: String,
    },
    /// Argumento de línea de órdenes o de configuración inválido.
    Config(String),
}

impl Error {
    /// Atajo para construir [`Error::Format`].
    pub fn format(what: &'static str, detail: impl Into<String>) -> Self {
        Self::Format {
            what,
            detail: detail.into(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "error de E/S: {e}"),
            Self::Format { what, detail } => write!(f, "{what} inválido: {detail}"),
            Self::Config(msg) => write!(f, "configuración inválida: {msg}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

/// Alias de resultado del crate.
pub type Result<T> = std::result::Result<T, Error>;
