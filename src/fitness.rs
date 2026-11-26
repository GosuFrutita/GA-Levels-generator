//! Función de adaptación multiobjetivo.
//!
//! La jugabilidad es una **restricción dura** (verificada por [`crate::play`]);
//! la calidad combina tres objetivos en `[0, 1]`: dificultad moderada (como
//! distancia a un objetivo), variedad estructural y coherencia visual.
//!
//! El escalar final es estrictamente positivo (lo exige la ruleta) y respeta el
//! orden lexicográfico *jugable ≻ injugable*:
//!
//! ```text
//! F = 0,1 + 0,9·Q            si el nivel es jugable       → F ∈ [0,1; 1]
//! F = 0,1·progreso + 1e-6    si no lo es                  → F ∈ (0; 0,1)
//! ```

use crate::level::Level;
use crate::metrics::{self, Features, MetricScratch, Targets};
use crate::play::{Checker, PlayResult, PlayScratch};

/// Forma de agregar los objetivos en un escalar.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Aggregation {
    /// Suma ponderada (Yang, cap. 14), con codificación factible/infactible.
    WeightedSum,
    /// Media geométrica ponderada: castiga cualquier objetivo cercano a 0.
    Geometric,
    /// Penalización estática ingenua: `Q − λ·(1 − progreso)`, sin restricción dura.
    Penalty(f64),
    /// Reglas de factibilidad de Deb (2000) dentro del torneo; el escalar es el de
    /// la suma ponderada (la ruleta lo usa tal cual).
    Deb,
}

/// Pesos de los objetivos.
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    /// Peso de la dificultad moderada.
    pub difficulty: f64,
    /// Peso de la variedad.
    pub variety: f64,
    /// Peso de la coherencia.
    pub coherence: f64,
    /// Peso de la naturalidad (modelo neuronal); 0 si no se usa.
    pub naturalness: f64,
}

impl Default for Weights {
    fn default() -> Self {
        Self {
            difficulty: 0.35,
            variety: 0.35,
            coherence: 0.30,
            naturalness: 0.0,
        }
    }
}

/// Configuración completa de la evaluación.
#[derive(Debug, Clone, PartialEq)]
pub struct FitnessConfig {
    /// Objetivos de las bandas.
    pub targets: Targets,
    /// Pesos.
    pub weights: Weights,
    /// Agregación.
    pub aggregation: Aggregation,
}

impl Default for FitnessConfig {
    fn default() -> Self {
        Self {
            targets: Targets::default(),
            weights: Weights::default(),
            aggregation: Aggregation::WeightedSum,
        }
    }
}

/// Resultado completo de evaluar un nivel.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Evaluation {
    /// Escalar a maximizar (siempre > 0).
    pub fitness: f64,
    /// ¿Es jugable?
    pub feasible: bool,
    /// Progreso horizontal alcanzado, `[0, 1]`.
    pub progress: f64,
    /// Calidad agregada `Q`, `[0, 1]`.
    pub quality: f64,
    /// Dificultad compuesta `D`.
    pub difficulty: f64,
    /// Puntuación de dificultad moderada `f_dif`.
    pub f_difficulty: f64,
    /// Variedad `V`.
    pub variety: f64,
    /// Coherencia `C`.
    pub coherence: f64,
    /// Encaje con el estilo.
    pub style_fit: f64,
    /// Naturalidad según el modelo neuronal (1 si no se usa).
    pub naturalness: f64,
    /// Rasgos medidos.
    pub features: Features,
}

impl Evaluation {
    /// Compara dos evaluaciones según las reglas de Deb: factible vence a
    /// infactible; entre factibles, mayor calidad; entre infactibles, mayor
    /// progreso (menor violación). Devuelve `true` si `self` gana a `other`.
    pub fn deb_beats(&self, other: &Self) -> bool {
        match (self.feasible, other.feasible) {
            (true, false) => true,
            (false, true) => false,
            (true, true) => self.quality > other.quality,
            (false, false) => self.progress > other.progress,
        }
    }
}

/// Puntuación externa opcional (p. ej. naturalidad de un modelo neuronal),
/// en `[0, 1]`. Debe ser `Sync` para evaluarse en paralelo.
pub trait Scorer: Sync {
    /// Puntúa un nivel.
    fn score(&self, level: &Level) -> f64;
}

/// Evaluador con memoria de trabajo propia (uno por hilo).
#[derive(Debug, Clone)]
pub struct Evaluator<'a> {
    checker: &'a Checker,
    config: &'a FitnessConfig,
    play: PlayScratch,
    metrics: MetricScratch,
}

impl<'a> Evaluator<'a> {
    /// Crea un evaluador.
    pub fn new(checker: &'a Checker, config: &'a FitnessConfig) -> Self {
        Self {
            checker,
            config,
            play: PlayScratch::default(),
            metrics: MetricScratch::default(),
        }
    }

    /// Ejecuta solo el verificador de jugabilidad.
    pub fn play(&mut self, level: &Level) -> PlayResult {
        self.checker.check(level, &mut self.play)
    }

    /// Evalúa un nivel por completo.
    pub fn evaluate(&mut self, level: &Level, scorer: Option<&dyn Scorer>) -> Evaluation {
        let play = self.checker.check(level, &mut self.play);
        let features = Features::compute(level, &play, &self.play, &mut self.metrics);
        let cfg = self.config;
        let t = &cfg.targets;
        let w = &cfg.weights;
        let max_jump = f64::from(self.checker.physics().jump_length);
        let d = metrics::difficulty(&features, max_jump);
        let f_dif = metrics::bell(d, t.difficulty, t.difficulty_sigma);
        let var = metrics::variety(&features, t);
        let coh = metrics::coherence(&features, level.style);
        let fit = metrics::style_fit(&features, level.style);
        let nat = match scorer {
            Some(s) if w.naturalness > 0.0 => s.score(level).clamp(0.0, 1.0),
            _ => 1.0,
        };
        let parts = [
            (w.difficulty, f_dif),
            (w.variety, var),
            (w.coherence, coh),
            (w.naturalness, nat),
        ];
        let wsum: f64 = parts.iter().map(|p| p.0).sum();
        let quality = match cfg.aggregation {
            Aggregation::Geometric => (parts
                .iter()
                .map(|&(wi, v)| wi * v.max(1e-6).ln())
                .sum::<f64>()
                / wsum)
                .exp(),
            _ => parts.iter().map(|&(wi, v)| wi * v).sum::<f64>() / wsum,
        };
        let feasible = play.solvable;
        let fitness = match cfg.aggregation {
            Aggregation::Penalty(lambda) => {
                // Sin restricción dura: el GA puede preferir un injugable "bonito".
                (quality - lambda * (1.0 - play.progress)).max(0.0) + 1e-6
            }
            _ if feasible => 0.1 + 0.9 * quality,
            _ => 0.1 * play.progress + 1e-6,
        };
        Evaluation {
            fitness,
            feasible,
            progress: play.progress,
            quality,
            difficulty: d,
            f_difficulty: f_dif,
            variety: var,
            coherence: coh,
            style_fit: fit,
            naturalness: nat,
            features,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level::{HEIGHT, Style, Tile};

    fn flat(width: usize, solvable: bool) -> Level {
        let mut l = Level::empty(width, Style::Base);
        for x in 0..width {
            if solvable || !(20..30).contains(&x) {
                l.set(x, HEIGHT - 1, Tile::Ground);
            }
        }
        l.set(1, HEIGHT - 2, Tile::Start);
        l.set(width - 2, HEIGHT - 2, Tile::Goal);
        l
    }

    #[test]
    fn jugable_siempre_supera_a_injugable_y_es_positivo() {
        let checker = Checker::default();
        for agg in [
            Aggregation::WeightedSum,
            Aggregation::Geometric,
            Aggregation::Deb,
        ] {
            let cfg = FitnessConfig {
                aggregation: agg,
                ..FitnessConfig::default()
            };
            let mut ev = Evaluator::new(&checker, &cfg);
            let good = ev.evaluate(&flat(64, true), None);
            let bad = ev.evaluate(&flat(64, false), None);
            assert!(good.feasible && !bad.feasible);
            assert!(good.fitness > bad.fitness, "{agg:?}");
            assert!(bad.fitness > 0.0 && good.fitness <= 1.0);
            assert!(good.deb_beats(&bad) && !bad.deb_beats(&good));
        }
    }
}
