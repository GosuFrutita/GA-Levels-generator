//! Redes neuronales hechas a mano: MLP con entrada categórica, pérdidas de
//! cross-entropy (categórica, binaria y de gradiente de política) y Adam.
//!
//! Este núcleo lo comparten el modelo de "naturalidad" de tiles, el agente de
//! aprendizaje por refuerzo y el modelo sustituto de jugabilidad.

pub mod adam;
pub mod float;
pub mod loss;
pub mod net;

pub use adam::Adam;
pub use float::Float;
pub use net::{Activation, Input, NONE, Net, NetSpec, Tape};
