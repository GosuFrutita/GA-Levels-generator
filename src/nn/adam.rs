//! Optimizador Adam (Kingma y Ba, 2015, <https://arxiv.org/abs/1412.6980>)
//! con recorte opcional de la norma global del gradiente.

use super::Float;

/// Estado de Adam para un vector de parámetros.
#[derive(Debug, Clone)]
pub struct Adam<T> {
    /// Tasa de aprendizaje.
    pub lr: f64,
    /// Decaimiento del primer momento.
    pub beta1: f64,
    /// Decaimiento del segundo momento.
    pub beta2: f64,
    /// Término de estabilidad.
    pub eps: f64,
    /// Norma máxima del gradiente (`None` = sin recorte).
    pub clip_norm: Option<f64>,
    m: Vec<T>,
    v: Vec<T>,
    t: i32,
}

impl<T: Float> Adam<T> {
    /// Adam con los valores por defecto del artículo (β₁ = 0,9, β₂ = 0,999, ε = 1e-8).
    pub fn new(n_params: usize, lr: f64) -> Self {
        Self {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            clip_norm: None,
            m: vec![T::ZERO; n_params],
            v: vec![T::ZERO; n_params],
            t: 0,
        }
    }

    /// Aplica un paso con el gradiente medio `grads` (no se modifica salvo recorte).
    /// Devuelve la norma del gradiente antes de recortar.
    pub fn step(&mut self, params: &mut [T], grads: &mut [T]) -> f64 {
        debug_assert_eq!(params.len(), grads.len());
        let norm = grads.iter().map(|g| g.to_f64().powi(2)).sum::<f64>().sqrt();
        if let Some(max) = self.clip_norm
            && norm > max
        {
            let k = T::from_f64(max / norm);
            for g in grads.iter_mut() {
                *g *= k;
            }
        }
        self.t += 1;
        let (b1, b2) = (T::from_f64(self.beta1), T::from_f64(self.beta2));
        let (c1, c2) = (T::ONE - b1, T::ONE - b2);
        // Corrección de sesgo incorporada en el paso (forma eficiente del artículo).
        let bc1 = 1.0 - self.beta1.powi(self.t);
        let bc2 = 1.0 - self.beta2.powi(self.t);
        let step = T::from_f64(self.lr * bc2.sqrt() / bc1);
        let eps = T::from_f64(self.eps * bc2.sqrt());
        for (((p, &g), m), v) in params
            .iter_mut()
            .zip(grads.iter())
            .zip(&mut self.m)
            .zip(&mut self.v)
        {
            *m = b1 * *m + c1 * g;
            *v = b2 * *v + c2 * g * g;
            *p -= step * *m / (v.sqrt() + eps);
        }
        norm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimiza_una_cuadratica() {
        let mut x = vec![3.0_f64, -2.0];
        let mut opt = Adam::new(2, 0.1);
        for _ in 0..500 {
            let mut g = vec![2.0 * (x[0] - 1.0), 2.0 * (x[1] + 0.5)];
            opt.step(&mut x, &mut g);
        }
        assert!(
            (x[0] - 1.0).abs() < 1e-3 && (x[1] + 0.5).abs() < 1e-3,
            "{x:?}"
        );
    }
}
