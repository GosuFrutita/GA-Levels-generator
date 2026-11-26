//! Selección adaptativa de operadores con un bandido multibrazo (aprendizaje
//! por refuerzo sin estado).
//!
//! *Dynamic Multi-Armed Bandit* (D-MAB; Da Costa, Fialho, Schoenauer y Sebag,
//! GECCO 2008): UCB1 elige el operador; la recompensa es la mejora de aptitud
//! del hijo frente al mejor padre; una prueba de Page-Hinkley detecta cambios
//! en la distribución de recompensas y reinicia las estadísticas (el mejor
//! operador cambia a lo largo de la evolución).
//!
//! Los valores por defecto de `c`, `delta` y `lambda` son de diseño propio: no
//! se pudieron leer en el artículo original y deben calibrarse.

/// Estado del bandido.
#[derive(Debug, Clone)]
pub struct Dmab {
    /// Coeficiente de exploración de UCB1.
    pub c: f64,
    /// Tolerancia de la prueba de Page-Hinkley.
    pub delta: f64,
    /// Umbral de la prueba de Page-Hinkley.
    pub lambda: f64,
    counts: Vec<f64>,
    means: Vec<f64>,
    ph_m: Vec<f64>,
    ph_max: Vec<f64>,
    total: f64,
    /// Número de reinicios provocados por Page-Hinkley.
    pub restarts: u32,
}

impl Dmab {
    /// Bandido de `arms` brazos.
    pub fn new(arms: usize) -> Self {
        Self {
            c: 0.5,
            delta: 0.05,
            lambda: 3.0,
            counts: vec![0.0; arms],
            means: vec![0.0; arms],
            ph_m: vec![0.0; arms],
            ph_max: vec![0.0; arms],
            total: 0.0,
            restarts: 0,
        }
    }

    /// Elige un brazo: primero prueba cada uno una vez; luego maximiza
    /// `media + c·√(2 ln N / n)`.
    pub fn choose(&self) -> usize {
        if let Some(i) = self.counts.iter().position(|&n| n == 0.0) {
            return i;
        }
        let ln_n = self.total.ln();
        (0..self.counts.len())
            .max_by(|&a, &b| {
                let ua = self.means[a] + self.c * (2.0 * ln_n / self.counts[a]).sqrt();
                let ub = self.means[b] + self.c * (2.0 * ln_n / self.counts[b]).sqrt();
                ua.total_cmp(&ub)
            })
            .unwrap_or(0)
    }

    /// Registra la recompensa `r` del brazo `arm`.
    pub fn update(&mut self, arm: usize, r: f64) {
        self.counts[arm] += 1.0;
        self.total += 1.0;
        let n = self.counts[arm];
        self.means[arm] += (r - self.means[arm]) / n;
        // Page-Hinkley: m_t acumula (media − r + δ); si max − m supera λ, la
        // recompensa ha caído de forma sostenida → reinicio.
        self.ph_m[arm] += self.means[arm] - r + self.delta;
        self.ph_max[arm] = self.ph_max[arm].max(self.ph_m[arm]);
        if self.ph_max[arm] - self.ph_m[arm] > self.lambda {
            self.reset();
            self.restarts += 1;
        }
    }

    fn reset(&mut self) {
        for v in [
            &mut self.counts,
            &mut self.means,
            &mut self.ph_m,
            &mut self.ph_max,
        ] {
            v.fill(0.0);
        }
        self.total = 0.0;
    }

    /// Frecuencia empírica de uso de cada brazo desde el último reinicio.
    pub fn usage(&self) -> Vec<f64> {
        let t = self.total.max(1.0);
        self.counts.iter().map(|c| c / t).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    #[test]
    fn encuentra_el_mejor_brazo() {
        let mut b = Dmab::new(3);
        b.lambda = f64::INFINITY; // sin reinicios para esta prueba
        let mut rng = Rng::seed_from_u64(1);
        let p = [0.2, 0.8, 0.4];
        let mut pulls = [0; 3];
        for _ in 0..3000 {
            let a = b.choose();
            pulls[a] += 1;
            b.update(a, f64::from(u8::from(rng.chance(p[a]))));
        }
        assert!(pulls[1] > pulls[0] && pulls[1] > pulls[2], "{pulls:?}");
    }
}
