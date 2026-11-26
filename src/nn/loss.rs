//! Funciones de pérdida y sus gradientes respecto a los *logits*.
//!
//! - Softmax con estabilización log-sum-exp.
//! - Cross-entropy categórica con suavizado de etiquetas y peso por muestra:
//!   `∂L/∂z = w · (p − y)`. Con `w = ventaja` es exactamente la pérdida de
//!   gradiente de política `−Â · log π(a|s)` (REINFORCE / A2C).
//! - Cross-entropy binaria con logits (estable): `∂L/∂z = σ(z) − y`.
//! - Entropía de la política y su gradiente, para el bono de exploración.

use super::Float;

/// `log Σ exp(z_i)` estable.
pub fn log_sum_exp<T: Float>(z: &[T]) -> T {
    let m = z.iter().copied().fold(z[0], T::max);
    let s = z.iter().fold(T::ZERO, |acc, &v| acc + (v - m).exp());
    m + s.ln()
}

/// Escribe `softmax(z)` en `p`.
pub fn softmax<T: Float>(z: &[T], p: &mut [T]) {
    let lse = log_sum_exp(z);
    for (pi, &zi) in p.iter_mut().zip(z) {
        *pi = (zi - lse).exp();
    }
}

/// Cross-entropy categórica de una muestra.
///
/// `smoothing` reparte esa masa uniformemente entre las `K` clases;
/// `weight` multiplica pérdida y gradiente (puede ser negativo: ventaja).
/// El gradiente respecto a los logits se **acumula** en `dz`.
/// Devuelve la pérdida sin ponderar (−log p suavizado).
pub fn cross_entropy<T: Float>(z: &[T], target: usize, smoothing: T, weight: T, dz: &mut [T]) -> T {
    debug_assert_eq!(z.len(), dz.len());
    let k = T::from_f64(z.len() as f64);
    let lse = log_sum_exp(z);
    let off = smoothing / k;
    let on = T::ONE - smoothing + off;
    let mut loss = T::ZERO;
    for (i, (&zi, d)) in z.iter().zip(dz.iter_mut()).enumerate() {
        let logp = zi - lse;
        let y = if i == target { on } else { off };
        loss -= y * logp;
        *d += weight * (logp.exp() - y);
    }
    loss
}

/// `−log p(target)` sin gradiente (para evaluar NLL / perplejidad).
pub fn nll<T: Float>(z: &[T], target: usize) -> T {
    log_sum_exp(z) - z[target]
}

/// Sigmoide estable.
pub fn sigmoid<T: Float>(z: T) -> T {
    if z >= T::ZERO {
        T::ONE / (T::ONE + (-z).exp())
    } else {
        let e = z.exp();
        e / (T::ONE + e)
    }
}

/// Cross-entropy binaria con logits: `max(z,0) − z·y + ln(1 + e^{−|z|})`.
/// Devuelve `(pérdida, ∂L/∂z)` ya multiplicados por `weight`.
pub fn bce_with_logits<T: Float>(z: T, y: T, weight: T) -> (T, T) {
    let abs = if z >= T::ZERO { z } else { -z };
    let loss = z.max(T::ZERO) - z * y + ((-abs).exp() + T::ONE).ln();
    (weight * loss, weight * (sigmoid(z) - y))
}

/// Entropía `H = −Σ p log p` de `softmax(z)`; acumula en `dz` el gradiente de
/// `−coef · H` (es decir, minimizar la pérdida **aumenta** la entropía).
pub fn entropy_bonus<T: Float>(z: &[T], coef: T, dz: &mut [T]) -> T {
    let lse = log_sum_exp(z);
    let mut h = T::ZERO;
    for &zi in z {
        let logp = zi - lse;
        h -= logp.exp() * logp;
    }
    // ∂H/∂z_i = −p_i (log p_i + H)  ⇒  ∂(−cH)/∂z_i = c · p_i (log p_i + H)
    for (&zi, d) in z.iter().zip(dz.iter_mut()) {
        let logp = zi - lse;
        *d += coef * logp.exp() * (logp + h);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numeric_grad(f: impl Fn(&[f64]) -> f64, z: &[f64]) -> Vec<f64> {
        let h = 1e-6;
        (0..z.len())
            .map(|i| {
                let mut a = z.to_vec();
                let mut b = z.to_vec();
                a[i] += h;
                b[i] -= h;
                (f(&a) - f(&b)) / (2.0 * h)
            })
            .collect()
    }

    #[test]
    fn softmax_suma_uno_y_es_estable() {
        let z = [1000.0_f64, 1001.0, 999.0];
        let mut p = [0.0; 3];
        softmax(&z, &mut p);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
        assert!(p.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn gradiente_cross_entropy() {
        let z = [0.3, -1.2, 2.0, 0.5];
        for &(t, s, w) in &[(2, 0.0, 1.0), (0, 0.1, 1.0), (1, 0.05, -0.7)] {
            let mut dz = [0.0; 4];
            cross_entropy(&z, t, s, w, &mut dz);
            let num = numeric_grad(
                |zz| {
                    let mut tmp = [0.0; 4];
                    w * cross_entropy(zz, t, s, 1.0, &mut tmp)
                },
                &z,
            );
            for (a, b) in dz.iter().zip(&num) {
                assert!((a - b).abs() < 1e-6, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn perdida_inicial_uniforme_es_ln_k() {
        let z = [0.0_f64; 6];
        assert!((nll(&z, 3) - 6.0_f64.ln()).abs() < 1e-12);
    }

    #[test]
    fn gradiente_bce() {
        for &(z, y) in &[(0.4, 1.0), (-3.0, 0.0), (25.0, 0.0), (-25.0, 1.0)] {
            let (_, d) = bce_with_logits(z, y, 1.0);
            let num = numeric_grad(|v| bce_with_logits(v[0], y, 1.0).0, &[z])[0];
            assert!((d - num).abs() < 1e-5, "{d} vs {num}");
        }
        assert!(bce_with_logits(800.0_f64, 0.0, 1.0).0.is_finite());
    }

    #[test]
    fn gradiente_entropia() {
        let z = [0.2, -0.4, 1.1];
        let mut dz = [0.0; 3];
        entropy_bonus(&z, 0.5, &mut dz);
        let num = numeric_grad(
            |zz| {
                let mut tmp = [0.0; 3];
                -0.5 * entropy_bonus(zz, 0.0, &mut tmp)
            },
            &z,
        );
        for (a, b) in dz.iter().zip(&num) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }
}
