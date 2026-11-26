//! Modelo de "naturalidad" de niveles entrenado con cross-entropy.
//!
//! Un MLP predice cada tile a partir de su contexto causal: una ventana
//! relativa en las dos columnas anteriores, los tiles ya vistos por debajo en la
//! columna actual y la fila absoluta. El orden de barrido va por columnas de
//! izquierda a derecha y dentro de cada columna de abajo arriba.
//!
//! Es la idea de los modelos secuenciales de Summerville y Mateas (2016) y de
//! las cadenas de Markov multidimensionales de Snodgrass y Ontañón, con un MLP
//! de contexto fijo al estilo de Bengio et al. (2003).
//!
//! La log-verosimilitud negativa (NLL) media por tile de un nivel evolucionado
//! se convierte en un término de aptitud en `[0, 1]`.
//!
//! Proyección del alfabeto: el corpus de Super Mario Bros no tiene escaleras ni
//! decoraciones, así que al puntuar se tratan como vacío (la decoración es solo
//! visual y la escalera no tiene equivalente). El modelo juzga la estructura.

use crate::error::{Error, Result};
use crate::fitness::Scorer;
use crate::level::{HEIGHT, Level, Tile};
use crate::nn::loss::{cross_entropy, nll};
use crate::nn::{Activation, Adam, Input, Net, NetSpec, Tape};
use crate::rng::Rng;
use std::fmt::Write as _;

/// Clases que el modelo predice.
pub const CLASSES: [Tile; 6] = [
    Tile::Empty,
    Tile::Ground,
    Tile::Platform,
    Tile::Hazard,
    Tile::Enemy,
    Tile::Coin,
];
/// Número de clases.
pub const K: usize = CLASSES.len();
/// Token de borde (fuera del nivel o aún no visto).
pub const BORDER: u16 = K as u16;
/// Columnas previas en el contexto.
pub const PREV_COLS: usize = 2;
/// Alcance vertical (±filas) de la ventana relativa en las columnas previas.
pub const REACH: usize = 6;
/// Tiles por debajo en la columna actual.
pub const BELOW: usize = 6;
/// Posiciones categóricas del contexto.
pub const POSITIONS: usize = PREV_COLS * (2 * REACH + 1) + BELOW;
/// Cota de la pérdida por tile: un patrón inédito no domina la media.
pub const MAX_TILE_NLL: f64 = 6.907_755_278_982_137; // −ln(1e-3)

/// Índice en el contexto de la celda de la izquierda a la misma altura.
const LEFT: usize = (PREV_COLS - 1) * (2 * REACH + 1) + REACH;
/// Índice en el contexto de la celda inmediatamente inferior.
const DOWN: usize = PREV_COLS * (2 * REACH + 1);

/// Clase de un tile tras la proyección del alfabeto.
pub fn class_of(t: Tile) -> u16 {
    match t {
        Tile::Ground => 1,
        Tile::Platform => 2,
        Tile::Hazard => 3,
        Tile::Enemy => 4,
        Tile::Coin => 5,
        // Vacío, escalera, decoración, salida y meta.
        _ => 0,
    }
}

/// Rellena el contexto causal del tile `(x, y)` con vecinos **relativos**
/// (como las cadenas de Markov multidimensionales): una ventana de ±[`REACH`]
/// filas en cada una de las [`PREV_COLS`] columnas previas y los [`BELOW`]
/// tiles de debajo en la columna actual. Fuera del nivel se usa [`BORDER`].
pub fn context(level: &Level, x: usize, y: usize, out: &mut [u16; POSITIONS]) {
    let cell = |cx: i32, cy: i32| {
        if cx < 0 || cy < 0 || cy >= HEIGHT as i32 {
            BORDER
        } else {
            class_of(level.get(cx as usize, cy as usize))
        }
    };
    let (xi, yi) = (x as i32, y as i32);
    let mut k = 0;
    for dx in (1..=PREV_COLS as i32).rev() {
        for dy in -(REACH as i32)..=REACH as i32 {
            out[k] = cell(xi - dx, yi + dy);
            k += 1;
        }
    }
    for dy in 1..=BELOW as i32 {
        out[k] = cell(xi, yi + dy);
        k += 1;
    }
}

/// Fila absoluta codificada one-hot (rasgo denso de la red).
pub fn row_one_hot(y: usize) -> [f32; HEIGHT] {
    let mut r = [0.0; HEIGHT];
    r[y] = 1.0;
    r
}

/// Una muestra de entrenamiento.
#[derive(Debug, Clone)]
pub struct Sample {
    /// Contexto.
    pub ctx: [u16; POSITIONS],
    /// Fila del tile.
    pub row: u8,
    /// Clase objetivo.
    pub target: u16,
}

/// Extrae todas las muestras de un conjunto de niveles.
pub fn samples(levels: &[Level]) -> Vec<Sample> {
    let mut out = Vec::new();
    for l in levels {
        for x in 0..l.width() {
            for y in 0..HEIGHT {
                let mut ctx = [0; POSITIONS];
                context(l, x, y, &mut ctx);
                out.push(Sample {
                    ctx,
                    row: y as u8,
                    target: class_of(l.get(x, y)),
                });
            }
        }
    }
    out
}

/// Línea base: frecuencias de clase (unigrama) con suavizado de Laplace.
pub fn unigram_nll(train: &[Sample], test: &[Sample]) -> f64 {
    let mut c = [1.0; K];
    for s in train {
        c[usize::from(s.target)] += 1.0;
    }
    let total: f64 = c.iter().sum();
    test.iter()
        .map(|s| -(c[usize::from(s.target)] / total).ln())
        .sum::<f64>()
        / test.len() as f64
}

/// Línea base: cadena de Markov 2D, `P(t | izquierda, debajo, fila)`, con
/// suavizado aditivo `α` (Snodgrass y Ontañón).
pub fn markov_nll(train: &[Sample], test: &[Sample], alpha: f64) -> f64 {
    let key = |s: &Sample| {
        (usize::from(s.ctx[LEFT]) * (K + 1) + usize::from(s.ctx[DOWN])) * HEIGHT
            + usize::from(s.row)
    };
    let mut counts = vec![[0.0; K]; (K + 1) * (K + 1) * HEIGHT];
    for s in train {
        counts[key(s)][usize::from(s.target)] += 1.0;
    }
    test.iter()
        .map(|s| {
            let row = &counts[key(s)];
            let total: f64 = row.iter().sum::<f64>() + alpha * K as f64;
            -((row[usize::from(s.target)] + alpha) / total).ln()
        })
        .sum::<f64>()
        / test.len() as f64
}

/// Hiperparámetros del entrenamiento.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Neuronas ocultas por capa.
    pub hidden: Vec<usize>,
    /// Épocas máximas.
    pub epochs: usize,
    /// Tamaño de mini-lote.
    pub batch: usize,
    /// Tasa de aprendizaje de Adam.
    pub lr: f64,
    /// Suavizado de etiquetas.
    pub smoothing: f64,
    /// Épocas sin mejora antes de parar.
    pub patience: usize,
    /// Semilla.
    pub seed: u64,
    /// Activación de las capas ocultas.
    pub activation: Activation,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            hidden: vec![32],
            epochs: 80,
            batch: 256,
            lr: 1e-3,
            smoothing: 0.0,
            patience: 5,
            seed: 0,
            // ReLU: la tangente hiperbólica dominaba el coste de puntuar niveles.
            activation: Activation::Relu,
        }
    }
}

/// Métricas de una época.
#[derive(Debug, Clone, Copy)]
pub struct EpochLog {
    /// Época.
    pub epoch: usize,
    /// NLL media de entrenamiento (con suavizado).
    pub train_loss: f64,
    /// NLL media de validación (sin suavizado).
    pub val_nll: f64,
    /// Exactitud de validación.
    pub val_acc: f64,
}

/// Modelo entrenado.
#[derive(Debug, Clone)]
pub struct NaturalnessModel {
    /// Red.
    pub net: Net<f32>,
    /// NLL de referencia (validación sobre el corpus).
    pub nll_ref: f64,
}

/// Evalúa NLL media y exactitud sobre un conjunto.
pub fn evaluate(net: &Net<f32>, data: &[Sample]) -> (f64, f64) {
    let mut tape = net.tape();
    let (mut loss, mut hits) = (0.0, 0);
    for s in data {
        net.forward(
            Input {
                cats: &s.ctx,
                dense: &row_one_hot(usize::from(s.row)),
            },
            &mut tape,
        );
        loss += f64::from(nll(&tape.logits, usize::from(s.target)));
        let pred = tape
            .logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map_or(0, |p| p.0);
        hits += usize::from(pred == usize::from(s.target));
    }
    (
        loss / data.len().max(1) as f64,
        hits as f64 / data.len().max(1) as f64,
    )
}

/// Entrena el modelo con Adam y parada temprana por NLL de validación.
pub fn train(
    train: &[Sample],
    val: &[Sample],
    cfg: &TrainConfig,
    mut log: impl FnMut(&EpochLog),
) -> (NaturalnessModel, Vec<EpochLog>) {
    let mut rng = Rng::seed_from_u64(cfg.seed);
    let spec = NetSpec {
        positions: POSITIONS,
        categories: K + 1,
        dense: HEIGHT,
        hidden: cfg.hidden.clone(),
        outputs: K,
        value_head: false,
        activation: cfg.activation,
    };
    let mut net: Net<f32> = Net::new(spec, &mut rng);
    let mut opt = Adam::new(net.num_params(), cfg.lr);
    let mut grads = vec![0.0_f32; net.num_params()];
    let mut tape: Tape<f32> = net.tape();
    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut history = Vec::new();
    let mut best = (f64::INFINITY, net.params.clone());
    let mut since_best = 0;
    let smoothing = cfg.smoothing as f32;
    for epoch in 1..=cfg.epochs {
        rng.shuffle(&mut order);
        let mut total = 0.0;
        for batch in order.chunks(cfg.batch) {
            grads.fill(0.0);
            let scale = 1.0 / batch.len() as f32;
            let mut dz = [0.0_f32; K];
            for &i in batch {
                let s = &train[i];
                net.forward(
                    Input {
                        cats: &s.ctx,
                        dense: &row_one_hot(usize::from(s.row)),
                    },
                    &mut tape,
                );
                dz.fill(0.0);
                total += f64::from(cross_entropy(
                    &tape.logits,
                    usize::from(s.target),
                    smoothing,
                    scale,
                    &mut dz,
                ));
                net.backward(&mut tape, &dz, 0.0, &mut grads);
            }
            opt.step(&mut net.params, &mut grads);
        }
        let (val_nll, val_acc) = evaluate(&net, val);
        let entry = EpochLog {
            epoch,
            train_loss: total / train.len() as f64,
            val_nll,
            val_acc,
        };
        log(&entry);
        history.push(entry);
        if val_nll < best.0 - 1e-4 {
            best = (val_nll, net.params.clone());
            since_best = 0;
        } else {
            since_best += 1;
            if since_best >= cfg.patience {
                break;
            }
        }
    }
    net.params = best.1;
    (
        NaturalnessModel {
            net,
            nll_ref: best.0,
        },
        history,
    )
}

impl NaturalnessModel {
    /// NLL media por tile de un nivel (cada tile acotado por [`MAX_TILE_NLL`]).
    pub fn level_nll(&self, level: &Level, tape: &mut Tape<f32>) -> f64 {
        let mut ctx = [0; POSITIONS];
        let mut total = 0.0;
        for x in 0..level.width() {
            for y in 0..HEIGHT {
                context(level, x, y, &mut ctx);
                self.net.forward(
                    Input {
                        cats: &ctx,
                        dense: &row_one_hot(y),
                    },
                    tape,
                );
                total += f64::from(nll(&tape.logits, usize::from(class_of(level.get(x, y)))))
                    .min(MAX_TILE_NLL);
            }
        }
        total / (level.width() * HEIGHT) as f64
    }

    /// Puntuación en `[0, 1]`: `NLL_ref / NLL`, que vale 1 si el nivel es tan
    /// "natural" como el corpus y decae suavemente (sin saturarse en 0) cuando
    /// se aleja de él, así el GA siempre recibe una señal útil.
    pub fn score_nll(&self, nll: f64) -> f64 {
        (self.nll_ref / nll.max(1e-12)).clamp(0.0, 1.0)
    }

    /// Serializa el modelo (NLL de referencia + red).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = b"NAT1".to_vec();
        out.extend_from_slice(&self.nll_ref.to_le_bytes());
        out.extend_from_slice(&self.net.to_bytes());
        out
    }

    /// Lee un modelo escrito con [`NaturalnessModel::to_bytes`].
    ///
    /// # Errors
    ///
    /// Si el formato es inválido.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let rest = bytes
            .strip_prefix(b"NAT1")
            .ok_or_else(|| Error::format("modelo de naturalidad", "firma incorrecta"))?;
        let (head, net) = rest
            .split_at_checked(8)
            .ok_or_else(|| Error::format("modelo de naturalidad", "truncado"))?;
        let mut b = [0; 8];
        b.copy_from_slice(head);
        Ok(Self {
            nll_ref: f64::from_le_bytes(b),
            net: Net::from_bytes(net)?,
        })
    }

    /// Informe de texto de un entrenamiento.
    pub fn report(history: &[EpochLog], baselines: &[(&str, f64)]) -> String {
        let mut s =
            String::from("época,pérdida_entrenamiento,nll_validación,exactitud_validación\n");
        for h in history {
            let _ = writeln!(
                s,
                "{},{:.5},{:.5},{:.4}",
                h.epoch, h.train_loss, h.val_nll, h.val_acc
            );
        }
        for (n, v) in baselines {
            let _ = writeln!(s, "# {n}: NLL {v:.5} (perplejidad {:.3})", v.exp());
        }
        s
    }
}

impl Scorer for NaturalnessModel {
    fn score(&self, level: &Level) -> f64 {
        let mut tape = self.net.tape();
        self.score_nll(self.level_nll(level, &mut tape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level::Style;

    fn toy(width: usize, rng: &mut Rng) -> Level {
        // Mundo de juguete: suelo de dos filas con huecos y algún enemigo.
        let mut l = Level::empty(width, Style::Base);
        for x in 0..width {
            if rng.chance(0.9) {
                l.set(x, HEIGHT - 1, Tile::Ground);
                l.set(x, HEIGHT - 2, Tile::Ground);
                if rng.chance(0.1) {
                    l.set(x, HEIGHT - 3, Tile::Enemy);
                }
            }
        }
        l
    }

    #[test]
    fn aprende_un_mundo_simple() {
        let mut rng = Rng::seed_from_u64(1);
        let tr: Vec<Level> = (0..6).map(|_| toy(60, &mut rng)).collect();
        let va: Vec<Level> = (0..2).map(|_| toy(60, &mut rng)).collect();
        let (st, sv) = (samples(&tr), samples(&va));
        let cfg = TrainConfig {
            epochs: 30,
            lr: 5e-3,
            hidden: vec![16],
            ..TrainConfig::default()
        };
        let (m, hist) = train(&st, &sv, &cfg, |_| {});
        let uni = unigram_nll(&st, &sv);
        assert!(m.nll_ref < uni, "MLP {} vs unigrama {uni}", m.nll_ref);
        assert!(hist.first().is_some_and(|h| h.val_nll > m.nll_ref - 1e-9));
        // Un nivel de ruido es menos natural que uno del mismo mundo.
        let mut noise = Level::empty(60, Style::Base);
        for t in noise.tiles_mut() {
            *t = CLASSES[rng.index(K)];
        }
        assert!(m.score(&va[0]) > m.score(&noise));
        let back = NaturalnessModel::from_bytes(&m.to_bytes()).expect("válido");
        assert!((back.score(&va[0]) - m.score(&va[0])).abs() < 1e-9);
    }
}
