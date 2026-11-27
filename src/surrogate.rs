//! Predicción de la jugabilidad **sin** ejecutar el verificador: rasgos
//! estructurales baratos, conjunto de datos etiquetado por el BFS y modelos de
//! referencia entrenados con cross-entropy binaria (regresión logística y MLP).
//! La Programación Genética ([`crate::gp`]) aprende sobre los mismos datos.
//!
//! Gestión del modelo sustituto (Jin, 2011): el BFS es barato, así que el
//! interés del sustituto aquí es metodológico (comparar un modelo simbólico
//! interpretable con uno conexionista), no ahorrar tiempo.

use crate::ga::init::random_level;
use crate::ga::variation::{MacroOp, macro_mutation, point_mutation};
use crate::level::{HEIGHT, Level, SAFE_COLS, Tile};
use crate::nn::loss::{bce_with_logits, sigmoid};
use crate::nn::{Activation, Adam, Input, Net, NetSpec};
use crate::play::{Checker, Physics, PlayScratch};
use crate::rng::Rng;

/// Número de rasgos.
pub const NF: usize = 21;

/// Nombres cortos de los rasgos (para imprimir expresiones).
pub const FEATURE_NAMES: [&str; NF] = [
    "huecos",
    "hueco_max",
    "hueco_medio",
    "huecos_largos",
    "subida_max",
    "subidas_altas",
    "enemigos",
    "peligros",
    "plataformas",
    "escaleras",
    "densidad",
    "altura_media",
    "altura_desv",
    "llano_max",
    "sin_suelo",
    "enemigos_borde",
    "peligros_suelo",
    "techo_bajo",
    "monedas",
    "vano_max",
    "vanos_largos",
];

/// Calcula los rasgos estructurales de un nivel (no usa el BFS).
pub fn features(level: &Level, physics: Physics) -> [f64; NF] {
    let w = level.width();
    let wf = w as f64;
    let jl = usize::from(physics.jump_length);
    let jh = i64::from(physics.jump_height);
    let tops: Vec<usize> = (0..w).map(|x| level.ground_top(x)).collect();
    let mut f = [0.0; NF];
    // Huecos (columnas sin suelo) y vanos (columnas sin ningún apoyo).
    let gaps = crate::metrics::column_runs(level, |x| tops[x] == HEIGHT);
    let spans =
        crate::metrics::column_runs(level, |x| !level.column(x).iter().any(|t| t.is_support()));
    let long = |runs: &[u32]| runs.iter().filter(|&&r| r as usize > jl).count() as f64;
    let n_gaps = gaps.len() as f64;
    f[0] = n_gaps / wf * 10.0;
    f[1] = f64::from(gaps.iter().copied().max().unwrap_or(0));
    f[2] = if gaps.is_empty() {
        0.0
    } else {
        f64::from(gaps.iter().sum::<u32>()) / n_gaps
    };
    f[3] = long(&gaps);
    // Subidas entre columnas consecutivas con suelo.
    let (mut rise_max, mut high_rises) = (0_i64, 0);
    for x in 1..w {
        if tops[x] < HEIGHT && tops[x - 1] < HEIGHT {
            let rise = tops[x - 1] as i64 - tops[x] as i64;
            rise_max = rise_max.max(rise);
            high_rises += usize::from(rise > jh);
        }
    }
    f[4] = rise_max as f64;
    f[5] = high_rises as f64;
    let count = |t: Tile| level.tiles().iter().filter(|&&v| v == t).count() as f64;
    f[6] = count(Tile::Enemy) / wf * 10.0;
    f[7] = count(Tile::Hazard) / wf * 10.0;
    f[8] = count(Tile::Platform) / wf;
    f[9] = count(Tile::Ladder) / wf * 10.0;
    f[10] = level.tiles().iter().filter(|t| t.is_solid()).count() as f64 / (wf * HEIGHT as f64);
    let heights: Vec<f64> = tops
        .iter()
        .filter(|&&t| t < HEIGHT)
        .map(|&t| (HEIGHT - t) as f64)
        .collect();
    let hm = heights.iter().sum::<f64>() / heights.len().max(1) as f64;
    f[11] = hm;
    f[12] = (heights.iter().map(|h| (h - hm).powi(2)).sum::<f64>() / heights.len().max(1) as f64)
        .sqrt();
    let (mut flat, mut flat_max) = (0, 0);
    for x in 1..w {
        if tops[x] == tops[x - 1] && tops[x] < HEIGHT {
            flat += 1;
            flat_max = flat_max.max(flat);
        } else {
            flat = 0;
        }
    }
    f[13] = f64::from(flat_max) / wf;
    f[14] = tops.iter().filter(|&&t| t == HEIGHT).count() as f64 / wf;
    // Enemigos junto a un hueco y peligros sobre la superficie.
    let (mut edge_enemies, mut ground_hazards, mut low_ceiling) = (0, 0, 0);
    for x in 0..w {
        let top = tops[x];
        if top < HEIGHT && top > 0 {
            let above = level.get(x, top - 1);
            let near_gap = (x > 0 && tops[x - 1] == HEIGHT) || (x + 1 < w && tops[x + 1] == HEIGHT);
            edge_enemies += usize::from(above == Tile::Enemy && near_gap);
            ground_hazards += usize::from(above == Tile::Hazard);
            low_ceiling +=
                usize::from((2..=3).any(|d| top >= d && level.get(x, top - d).is_solid()));
        }
    }
    f[15] = edge_enemies as f64;
    f[16] = ground_hazards as f64 / wf * 10.0;
    f[17] = low_ceiling as f64 / wf * 10.0;
    f[18] = count(Tile::Coin) / wf;
    f[19] = f64::from(spans.iter().copied().max().unwrap_or(0));
    f[20] = long(&spans);
    f
}

/// Conjunto de datos: rasgos y etiqueta (1 = jugable).
#[derive(Debug, Clone, Default)]
pub struct Dataset {
    /// Rasgos por muestra.
    pub x: Vec<[f64; NF]>,
    /// Etiquetas.
    pub y: Vec<f64>,
}

impl Dataset {
    /// Número de muestras.
    pub fn len(&self) -> usize {
        self.y.len()
    }

    /// ¿Está vacío?
    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }

    /// Fracción de positivos.
    pub fn positive_rate(&self) -> f64 {
        self.y.iter().sum::<f64>() / self.len().max(1) as f64
    }

    /// Divide en entrenamiento y prueba (una de cada `k` muestras a prueba).
    pub fn split(&self, k: usize) -> (Self, Self) {
        let (mut a, mut b) = (Self::default(), Self::default());
        for (i, (x, y)) in self.x.iter().zip(&self.y).enumerate() {
            let d = if i % k == k - 1 { &mut b } else { &mut a };
            d.x.push(*x);
            d.y.push(*y);
        }
        (a, b)
    }

    /// Media y desviación de cada rasgo (para estandarizar).
    pub fn moments(&self) -> ([f64; NF], [f64; NF]) {
        let n = self.len().max(1) as f64;
        let mut m = [0.0; NF];
        let mut s = [0.0; NF];
        for x in &self.x {
            for (mi, xi) in m.iter_mut().zip(x) {
                *mi += xi / n;
            }
        }
        for x in &self.x {
            for i in 0..NF {
                s[i] += (x[i] - m[i]).powi(2) / n;
            }
        }
        (m, s.map(|v| v.sqrt().max(1e-9)))
    }

    /// Estandariza con momentos dados.
    pub fn standardize(&mut self, m: &[f64; NF], s: &[f64; NF]) {
        for x in &mut self.x {
            for i in 0..NF {
                x[i] = (x[i] - m[i]) / s[i];
            }
        }
    }
}

/// Genera un conjunto de datos etiquetado por el BFS con clases equilibradas:
/// niveles aleatorios y variantes mutadas de niveles evolucionados (con más
/// mutaciones, más probable que dejen de ser jugables).
pub fn build_dataset(
    n: usize,
    width: usize,
    evolved: &[Level],
    checker: &Checker,
    seed: u64,
) -> Dataset {
    let mut rng = Rng::seed_from_u64(seed);
    let mut scratch = PlayScratch::default();
    let mut d = Dataset::default();
    // Clases equilibradas: como mucho la mitad de cada etiqueta.
    let mut per_class = [0_usize; 2];
    let mut i = 0_usize;
    while d.len() < n {
        i += 1;
        let level = if evolved.is_empty() || i.is_multiple_of(2) {
            random_level(width, &mut rng)
        } else {
            let mut l = evolved[rng.index(evolved.len())].clone();
            for _ in 0..rng.index(8) {
                macro_mutation(
                    &mut l,
                    MacroOp::ALL[rng.index(MacroOp::ALL.len())],
                    &mut rng,
                );
            }
            let cells = ((l.width() - 2 * SAFE_COLS) * HEIGHT) as f64;
            point_mutation(&mut l, rng.f64() * 20.0 / cells, &mut rng);
            crate::ga::init::repair(&mut l, true);
            l
        };
        let label = usize::from(checker.check(&level, &mut scratch).solvable);
        if per_class[label] * 2 >= n {
            continue;
        }
        per_class[label] += 1;
        d.x.push(features(&level, checker.physics()));
        d.y.push(label as f64);
    }
    // Se baraja porque el orden de generación alterna orígenes y clases
    // mientras que la partición en entrenamiento y prueba es por posición.
    let mut order: Vec<usize> = (0..d.len()).collect();
    rng.shuffle(&mut order);
    Dataset {
        x: order.iter().map(|&k| d.x[k]).collect(),
        y: order.iter().map(|&k| d.y[k]).collect(),
    }
}

/// Métricas de un clasificador probabilístico.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClassMetrics {
    /// BCE media.
    pub bce: f64,
    /// Exactitud con umbral 0,5.
    pub accuracy: f64,
    /// F1 de la clase "injugable".
    pub f1_unplayable: f64,
    /// Área bajo la curva ROC.
    pub auc: f64,
    /// Error de calibración esperado (10 intervalos; Guo et al., 2017).
    pub ece: f64,
}

/// Calcula las métricas a partir de probabilidades y etiquetas.
pub fn metrics(p: &[f64], y: &[f64]) -> ClassMetrics {
    let n = p.len().max(1) as f64;
    let bce = p
        .iter()
        .zip(y)
        .map(|(&pi, &yi)| -(yi * pi.max(1e-12).ln() + (1.0 - yi) * (1.0 - pi).max(1e-12).ln()))
        .sum::<f64>()
        / n;
    let acc = p
        .iter()
        .zip(y)
        .filter(|(pi, yi)| (**pi >= 0.5) == (**yi >= 0.5))
        .count() as f64
        / n;
    // F1 de la clase negativa (injugable).
    let (mut tp, mut fp, mut fneg) = (0.0, 0.0, 0.0);
    for (&pi, &yi) in p.iter().zip(y) {
        match (pi < 0.5, yi < 0.5) {
            (true, true) => tp += 1.0,
            (true, false) => fp += 1.0,
            (false, true) => fneg += 1.0,
            _ => {}
        }
    }
    let f1 = if tp == 0.0 {
        0.0
    } else {
        2.0 * tp / (2.0 * tp + fp + fneg)
    };
    // AUC = P(score positivo > score negativo) = Â12 entre los dos grupos.
    let pos: Vec<f64> = p
        .iter()
        .zip(y)
        .filter(|t| *t.1 >= 0.5)
        .map(|t| *t.0)
        .collect();
    let neg: Vec<f64> = p
        .iter()
        .zip(y)
        .filter(|t| *t.1 < 0.5)
        .map(|t| *t.0)
        .collect();
    let auc = crate::stats::a12(&pos, &neg);
    // ECE con 10 intervalos de confianza de la predicción "jugable".
    let mut bins = [(0.0, 0.0, 0.0); 10];
    for (&pi, &yi) in p.iter().zip(y) {
        let b = ((pi * 10.0) as usize).min(9);
        bins[b].0 += 1.0;
        bins[b].1 += pi;
        bins[b].2 += yi;
    }
    let ece = bins
        .iter()
        .filter(|b| b.0 > 0.0)
        .map(|b| b.0 / n * (b.1 / b.0 - b.2 / b.0).abs())
        .sum();
    ClassMetrics {
        bce,
        accuracy: acc,
        f1_unplayable: f1,
        auc,
        ece,
    }
}

/// Regresión logística entrenada con BCE y Adam.
#[derive(Debug, Clone)]
pub struct Logistic {
    /// Pesos (`NF`) y sesgo al final.
    pub w: Vec<f64>,
}

impl Logistic {
    /// Entrena por épocas con mini-lotes.
    pub fn train(d: &Dataset, epochs: usize, lr: f64, seed: u64) -> Self {
        let mut rng = Rng::seed_from_u64(seed);
        let mut w = vec![0.0; NF + 1];
        let mut opt = Adam::new(NF + 1, lr);
        let mut g = vec![0.0; NF + 1];
        let mut order: Vec<usize> = (0..d.len()).collect();
        for _ in 0..epochs {
            rng.shuffle(&mut order);
            for batch in order.chunks(128) {
                g.fill(0.0);
                for &i in batch {
                    let z = w[NF] + d.x[i].iter().zip(&w).map(|(a, b)| a * b).sum::<f64>();
                    let (_, dz) = bce_with_logits(z, d.y[i], 1.0 / batch.len() as f64);
                    for k in 0..NF {
                        g[k] += dz * d.x[i][k];
                    }
                    g[NF] += dz;
                }
                opt.step(&mut w, &mut g);
            }
        }
        Self { w }
    }

    /// Probabilidad de que sea jugable.
    pub fn predict(&self, x: &[f64; NF]) -> f64 {
        sigmoid(self.w[NF] + x.iter().zip(&self.w).map(|(a, b)| a * b).sum::<f64>())
    }
}

/// MLP sustituto entrenado con BCE.
pub fn train_mlp(d: &Dataset, hidden: usize, epochs: usize, lr: f64, seed: u64) -> Net<f32> {
    let mut rng = Rng::seed_from_u64(seed);
    let spec = NetSpec {
        positions: 0,
        categories: 1,
        dense: NF,
        hidden: vec![hidden],
        outputs: 1,
        value_head: false,
        activation: Activation::Relu,
    };
    let mut net: Net<f32> = Net::new(spec, &mut rng);
    let mut opt = Adam::new(net.num_params(), lr);
    let mut g = vec![0.0_f32; net.num_params()];
    let mut tape = net.tape();
    let mut order: Vec<usize> = (0..d.len()).collect();
    for _ in 0..epochs {
        rng.shuffle(&mut order);
        for batch in order.chunks(128) {
            g.fill(0.0);
            for &i in batch {
                let x = d.x[i].map(|v| v as f32);
                net.forward(
                    Input {
                        cats: &[],
                        dense: &x,
                    },
                    &mut tape,
                );
                let (_, dz) =
                    bce_with_logits(tape.logits[0], d.y[i] as f32, 1.0 / batch.len() as f32);
                net.backward(&mut tape, &[dz], 0.0, &mut g);
            }
            opt.step(&mut net.params, &mut g);
        }
    }
    net
}

/// Probabilidad de jugable según el MLP.
pub fn mlp_predict(net: &Net<f32>, x: &[f64; NF], tape: &mut crate::nn::Tape<f32>) -> f64 {
    let xf = x.map(|v| v as f32);
    net.forward(
        Input {
            cats: &[],
            dense: &xf,
        },
        tape,
    );
    f64::from(sigmoid(tape.logits[0]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rasgos_de_un_hueco_infranqueable() {
        let mut l = Level::empty(40, crate::level::Style::Base);
        for x in 0..40 {
            if !(15..25).contains(&x) {
                l.set(x, HEIGHT - 1, Tile::Ground);
            }
        }
        let f = features(&l, Physics::default());
        assert!((f[1] - 10.0).abs() < 1e-12, "hueco máximo {}", f[1]);
        assert!((f[3] - 1.0).abs() < 1e-12);
        assert!((f[19] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn metricas_perfectas_y_aleatorias() {
        let y = [1.0, 0.0, 1.0, 0.0];
        let m = metrics(&[0.9, 0.1, 0.8, 0.2], &y);
        assert!((m.accuracy - 1.0).abs() < 1e-12 && (m.auc - 1.0).abs() < 1e-12);
        let m = metrics(&[0.5; 4], &y);
        assert!((m.auc - 0.5).abs() < 1e-12);
    }

    #[test]
    fn la_logistica_aprende_una_regla_lineal() {
        let mut d = Dataset::default();
        let mut rng = Rng::seed_from_u64(1);
        for _ in 0..2000 {
            let mut x = [0.0; NF];
            x[1] = rng.normal();
            d.y.push(f64::from(u8::from(x[1] < 0.3)));
            d.x.push(x);
        }
        let m = Logistic::train(&d, 30, 0.05, 0);
        let p: Vec<f64> = d.x.iter().map(|x| m.predict(x)).collect();
        assert!(metrics(&p, &d.y).accuracy > 0.95);
    }
}
