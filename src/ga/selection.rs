//! Selección de padres: torneo y ruleta (con escalado y muestreo universal
//! estocástico). Devuelven **índices**, nunca copias de individuos.

use crate::rng::Rng;

/// Escalado de la aptitud para la ruleta.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scaling {
    /// Aptitud cruda: `p_i = f_i / Σf`.
    None,
    /// Escalado lineal de Goldberg (1989): `f' = a·f + b` con `f'_media = f_media`
    /// y `f'_max = c·f_media`; si el mínimo escalado sería negativo, se ajusta
    /// para que `f'_min = 0`.
    Linear(f64),
    /// Truncamiento sigma: `f' = max(0, f − (f_media − c·σ))`.
    Sigma(f64),
}

/// Método de selección.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Selection {
    /// Torneo de tamaño `k` (con reemplazo).
    Tournament(usize),
    /// Ruleta proporcional a la aptitud.
    Roulette {
        /// Escalado previo.
        scaling: Scaling,
        /// `true`: muestreo universal estocástico (Baker, 1987);
        /// `false`: `n` giros independientes.
        sus: bool,
    },
}

impl Selection {
    /// Nombre para informes.
    pub fn name(self) -> String {
        match self {
            Self::Tournament(k) => format!("torneo-{k}"),
            Self::Roulette { scaling, sus } => {
                let s = match scaling {
                    Scaling::None => "cruda".to_string(),
                    Scaling::Linear(c) => format!("lineal-{c}"),
                    Scaling::Sigma(c) => format!("sigma-{c}"),
                };
                format!("ruleta-{s}{}", if sus { "-sus" } else { "" })
            }
        }
    }
}

/// Aplica el escalado y escribe las aptitudes escaladas (no negativas) en `out`.
pub fn scale(fitness: &[f64], scaling: Scaling, out: &mut Vec<f64>) {
    out.clear();
    let n = fitness.len() as f64;
    let mean = fitness.iter().sum::<f64>() / n;
    let max = fitness.iter().copied().fold(f64::MIN, f64::max);
    let min = fitness.iter().copied().fold(f64::MAX, f64::min);
    match scaling {
        Scaling::None => out.extend_from_slice(fitness),
        Scaling::Linear(c) => {
            let (a, b) = if (max - mean).abs() < 1e-12 {
                (1.0, 0.0)
            } else if min > (c * mean - max) / (c - 1.0) {
                // Caso normal: f'_max = c · f_media.
                let d = max - mean;
                ((c - 1.0) * mean / d, mean * (max - c * mean) / d)
            } else {
                // El mínimo quedaría negativo: se escala para que f'_min = 0.
                let d = mean - min;
                (mean / d, -min * mean / d)
            };
            out.extend(fitness.iter().map(|&f| (a * f + b).max(0.0)));
        }
        Scaling::Sigma(c) => {
            let sd = (fitness.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / n).sqrt();
            out.extend(fitness.iter().map(|&f| (f - (mean - c * sd)).max(0.0)));
        }
    }
    // Si todo es cero (población uniforme), la ruleta pasa a ser uniforme.
    if out.iter().all(|&v| v <= 0.0) {
        out.fill(1.0);
    }
}

/// Ruleta con `n` giros independientes: suma de prefijos + búsqueda binaria
/// (`O(log N)` por giro).
pub fn roulette(
    weights: &[f64],
    n: usize,
    rng: &mut Rng,
    prefix: &mut Vec<f64>,
    out: &mut Vec<u32>,
) {
    prefix.clear();
    let mut acc = 0.0;
    for &w in weights {
        acc += w;
        prefix.push(acc);
    }
    for _ in 0..n {
        let r = rng.f64() * acc;
        let i = prefix.partition_point(|&p| p <= r).min(weights.len() - 1);
        out.push(i as u32);
    }
}

/// Muestreo universal estocástico: un solo número aleatorio y `n` punteros
/// equiespaciados (`O(N)` en total, varianza mínima). El resultado sale ordenado,
/// así que el llamador debe barajarlo antes de emparejar.
pub fn sus(weights: &[f64], n: usize, rng: &mut Rng, out: &mut Vec<u32>) {
    let total: f64 = weights.iter().sum();
    let step = total / n as f64;
    let mut ptr = rng.f64() * step;
    let mut acc = 0.0;
    let mut i = 0;
    for _ in 0..n {
        while i + 1 < weights.len() && acc + weights[i] <= ptr {
            acc += weights[i];
            i += 1;
        }
        out.push(i as u32);
        ptr += step;
    }
}

/// Torneo de tamaño `k`: `better(a, b)` indica si `a` gana a `b`.
pub fn tournament(
    n_pop: usize,
    k: usize,
    n: usize,
    rng: &mut Rng,
    better: impl Fn(usize, usize) -> bool,
    out: &mut Vec<u32>,
) {
    for _ in 0..n {
        let mut best = rng.index(n_pop);
        for _ in 1..k {
            let c = rng.index(n_pop);
            if better(c, best) {
                best = c;
            }
        }
        out.push(best as u32);
    }
}

/// Intensidad de selección teórica del torneo (aproximación de Blickle y
/// Thiele, 1995): `I(t) ≈ √(2(ln t − ln √(4,14 ln t)))`.
pub fn tournament_intensity(t: usize) -> f64 {
    if t < 2 {
        return 0.0;
    }
    let lt = (t as f64).ln();
    (2.0 * (lt - (4.14 * lt).sqrt().ln())).sqrt()
}

/// Pérdida de diversidad teórica del torneo (Blickle y Thiele, 1995):
/// `t^(−1/(t−1)) − t^(−t/(t−1))`.
pub fn tournament_loss_of_diversity(t: usize) -> f64 {
    if t < 2 {
        return 0.0;
    }
    let tf = t as f64;
    tf.powf(-1.0 / (tf - 1.0)) - tf.powf(-tf / (tf - 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn escalado_lineal_conserva_media_y_fija_maximo() {
        let f = [1.0, 2.0, 3.0, 4.0, 10.0];
        let mut out = Vec::new();
        scale(&f, Scaling::Linear(1.5), &mut out);
        let mean = f.iter().sum::<f64>() / 5.0;
        let smean = out.iter().sum::<f64>() / 5.0;
        assert!((mean - smean).abs() < 1e-9);
        assert!((out[4] - 1.5 * mean).abs() < 1e-9);
        // Caso con mínimo negativo: se corrige a 0.
        let f = [0.01, 1.0, 1.0, 1.0, 1.0];
        scale(&f, Scaling::Linear(2.0), &mut out);
        assert!(out.iter().all(|&v| v >= 0.0));
        assert!(out[0].abs() < 1e-9);
    }

    #[test]
    fn sus_es_casi_exacto() {
        let w = [1.0, 2.0, 3.0, 4.0];
        let mut rng = Rng::seed_from_u64(1);
        let mut out = Vec::new();
        sus(&w, 100, &mut rng, &mut out);
        let mut counts = [0; 4];
        for &i in &out {
            counts[i as usize] += 1;
        }
        // Cada individuo recibe ⌊esperado⌋ o ⌈esperado⌉ copias.
        for (c, e) in counts.iter().zip([10.0, 20.0, 30.0, 40.0]) {
            assert!((f64::from(*c) - e).abs() <= 1.0, "{counts:?}");
        }
    }

    #[test]
    fn ruleta_proporcional() {
        let w = [1.0, 3.0];
        let mut rng = Rng::seed_from_u64(2);
        let (mut prefix, mut out) = (Vec::new(), Vec::new());
        roulette(&w, 40_000, &mut rng, &mut prefix, &mut out);
        let ones = out.iter().filter(|&&i| i == 1).count() as f64 / 40_000.0;
        assert!((ones - 0.75).abs() < 0.01, "{ones}");
    }

    #[test]
    fn intensidad_coincide_con_blickle_thiele() {
        // Tabla 8.3 del informe TIK 11: I(2)=0,56, I(3)=0,84, I(5)=1,16, I(10)=1,54.
        for (t, i) in [(2, 0.56), (3, 0.84), (5, 1.16), (10, 1.54)] {
            assert!(
                (tournament_intensity(t) - i).abs() < 0.02,
                "t={t}: {}",
                tournament_intensity(t)
            );
        }
        assert!((tournament_loss_of_diversity(2) - 0.25).abs() < 1e-12);
        assert!((tournament_loss_of_diversity(3) - 0.385).abs() < 1e-3);
    }
}
