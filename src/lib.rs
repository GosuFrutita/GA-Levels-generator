//! # nivelga
//!
//! Generación evolutiva de niveles de plataformas 2D: un algoritmo genético
//! evoluciona la matriz de tipos lógicos de tiles y el gen que elige el tileset
//! (uno de los cuatro packs *Pixel Platformer* de Kenney, CC0).
//!
//! Todo está implementado a mano, sin dependencias externas: generador
//! aleatorio, PNG (DEFLATE incluido), verificador de jugabilidad, métricas,
//! operadores genéticos, estadística y visualización.

pub mod ansi;
pub mod baselines;
pub mod cli;
pub mod error;
pub mod experiment;
pub mod fitness;
pub mod ga;
pub mod image;
pub mod level;
pub mod metrics;
pub mod naturalness;
pub mod nn;
pub mod play;
pub mod plot;
pub mod render;
pub mod report;
pub mod rng;
pub mod stats;
pub mod vglc;

pub use error::{Error, Result};
