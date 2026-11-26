//! Líneas base para comparar con el algoritmo genético con el mismo
//! presupuesto de evaluaciones: búsqueda aleatoria y (1+1)-EA.
//!
//! Ambas consumen las evaluaciones en "generaciones" del mismo tamaño que el
//! GA (`N` al inicio y `N − E` después), así las curvas de convergencia se
//! comparan punto a punto.

use crate::fitness::{Evaluation, Evaluator, Scorer};
use crate::ga::{GaConfig, init, variation};
use crate::level::{HEIGHT, Level, SAFE_COLS};
use crate::play::Checker;
use crate::rng::Rng;
use std::time::Instant;

/// Resultado de una línea base.
#[derive(Debug, Clone)]
pub struct BaselineResult {
    /// Mejor aptitud histórica al final de cada "generación".
    pub curve: Vec<f64>,
    /// Evaluaciones acumuladas al final de cada "generación".
    pub evaluations: Vec<usize>,
    /// Mejor nivel.
    pub best: Level,
    /// Su evaluación.
    pub best_eval: Evaluation,
    /// Evaluaciones hasta el primer nivel jugable (si lo hubo).
    pub first_feasible: Option<usize>,
    /// Segundos de ejecución.
    pub seconds: f64,
}

struct Tracker {
    best: Option<(Level, Evaluation)>,
    evals: usize,
    first_feasible: Option<usize>,
    curve: Vec<f64>,
    marks: Vec<usize>,
}

impl Tracker {
    fn offer(&mut self, l: &Level, e: Evaluation) {
        self.evals += 1;
        if e.feasible && self.first_feasible.is_none() {
            self.first_feasible = Some(self.evals);
        }
        if self
            .best
            .as_ref()
            .is_none_or(|(_, b)| e.fitness > b.fitness)
        {
            self.best = Some((l.clone(), e));
        }
    }

    fn mark(&mut self) {
        self.curve
            .push(self.best.as_ref().map_or(0.0, |b| b.1.fitness));
        self.marks.push(self.evals);
    }

    fn finish(self, t0: Instant) -> BaselineResult {
        let (best, best_eval) = self.best.unwrap_or_else(|| {
            (
                Level::empty(2 * SAFE_COLS + 1, crate::level::Style::Base),
                Evaluation::default(),
            )
        });
        BaselineResult {
            curve: self.curve,
            evaluations: self.marks,
            best,
            best_eval,
            first_feasible: self.first_feasible,
            seconds: t0.elapsed().as_secs_f64(),
        }
    }
}

fn tracker() -> Tracker {
    Tracker {
        best: None,
        evals: 0,
        first_feasible: None,
        curve: Vec::new(),
        marks: Vec::new(),
    }
}

/// Búsqueda aleatoria con el generador constructivo de la población inicial.
pub fn random_search(
    cfg: &GaConfig,
    checker: &Checker,
    scorer: Option<&dyn Scorer>,
) -> BaselineResult {
    let t0 = Instant::now();
    let mut rng = Rng::seed_from_u64(cfg.seed);
    let mut ev = Evaluator::new(checker, &cfg.fitness);
    let mut t = tracker();
    for g in 0..=cfg.generations {
        let batch = if g == 0 {
            cfg.population
        } else {
            cfg.population - cfg.elitism
        };
        for _ in 0..batch {
            let l = init::random_level(cfg.width, &mut rng);
            let e = ev.evaluate(&l, scorer);
            t.offer(&l, e);
        }
        t.mark();
    }
    t.finish(t0)
}

/// (1+1)-EA: un padre, un hijo por evaluación con los mismos operadores de
/// mutación que el GA; el hijo reemplaza al padre si no es peor.
pub fn one_plus_one(
    cfg: &GaConfig,
    checker: &Checker,
    scorer: Option<&dyn Scorer>,
) -> BaselineResult {
    let t0 = Instant::now();
    let mut rng = Rng::seed_from_u64(cfg.seed);
    let mut ev = Evaluator::new(checker, &cfg.fitness);
    let mut t = tracker();
    let cells = ((cfg.width - 2 * SAFE_COLS) * HEIGHT) as f64;
    let mut parent = init::random_level(cfg.width, &mut rng);
    let mut parent_eval = ev.evaluate(&parent, scorer);
    t.offer(&parent, parent_eval.clone());
    let mut child = parent.clone();
    for g in 0..=cfg.generations {
        let batch = if g == 0 {
            cfg.population - 1
        } else {
            cfg.population - cfg.elitism
        };
        for _ in 0..batch {
            child.clone_from(&parent);
            variation::point_mutation(&mut child, cfg.mutations_per_child / cells, &mut rng);
            if rng.chance(cfg.p_macro) {
                let op = variation::MacroOp::ALL[rng.index(variation::MacroOp::ALL.len())];
                variation::macro_mutation(&mut child, op, &mut rng);
            }
            if rng.chance(cfg.p_style) {
                variation::style_mutation(&mut child, &mut rng);
            }
            init::repair(&mut child, cfg.repair);
            let e = ev.evaluate(&child, scorer);
            t.offer(&child, e.clone());
            if e.fitness >= parent_eval.fitness {
                std::mem::swap(&mut parent, &mut child);
                parent_eval = e;
            }
        }
        t.mark();
    }
    t.finish(t0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismo_presupuesto_que_el_ga() {
        let cfg = GaConfig {
            population: 20,
            generations: 5,
            width: 50,
            elitism: 2,
            ..GaConfig::default()
        };
        let checker = Checker::default();
        let budget = 20 + 5 * 18;
        let r = random_search(&cfg, &checker, None);
        assert_eq!(r.evaluations.last(), Some(&budget));
        let o = one_plus_one(&cfg, &checker, None);
        assert_eq!(o.evaluations.last(), Some(&budget));
        // La curva del mejor histórico es monótona.
        assert!(o.curve.windows(2).all(|w| w[1] >= w[0]));
    }
}
