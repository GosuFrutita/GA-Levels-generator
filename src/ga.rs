//! Algoritmo genético generacional con elitismo.
//!
//! Bucle por generación:
//! 1. Copia los `E` mejores (elitismo) al búfer siguiente.
//! 2. Selecciona `N − E` padres (torneo o ruleta) como **índices**.
//! 3. Empareja, cruza con probabilidad `p_c` y muta cada hijo.
//! 4. Repara y evalúa los hijos en paralelo (`std::thread::scope`).
//! 5. Intercambia los búferes (`std::mem::swap`, sin asignar memoria).
//!
//! Selección, cruce y mutación se ejecutan en serie con el generador maestro;
//! la evaluación es determinista, así que el resultado es idéntico con
//! cualquier número de hilos.

pub mod adaptive;
pub mod init;
pub mod selection;
pub mod variation;

use crate::fitness::{Aggregation, Evaluation, Evaluator, FitnessConfig, Scorer};
use crate::level::{HEIGHT, Level, SAFE_COLS, Style, Tile};
use crate::play::Checker;
use crate::rng::Rng;
use adaptive::Dmab;
use selection::{Selection, scale};
use std::time::Instant;
use variation::{Crossover, MacroOp, Mutation};

/// Control de los operadores macro.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorControl {
    /// Operador macro uniforme al azar.
    Uniform,
    /// Bandido D-MAB (aprendizaje por refuerzo).
    Bandit,
}

/// Configuración del algoritmo genético.
#[derive(Debug, Clone, PartialEq)]
pub struct GaConfig {
    /// Semilla.
    pub seed: u64,
    /// Tamaño de población `N`.
    pub population: usize,
    /// Generaciones `G`.
    pub generations: usize,
    /// Ancho de los niveles.
    pub width: usize,
    /// Método de selección.
    pub selection: Selection,
    /// Operador de cruce.
    pub crossover: Crossover,
    /// Probabilidad de cruce por pareja.
    pub p_crossover: f64,
    /// Mutación puntual expresada como cambios esperados por hijo (`k/L`).
    pub mutations_per_child: f64,
    /// Probabilidad de mutación macro por hijo.
    pub p_macro: f64,
    /// Probabilidad de mutar el gen de estilo por hijo.
    pub p_style: f64,
    /// Número de élites.
    pub elitism: usize,
    /// Reparación completa tras la variación.
    pub repair: bool,
    /// Control de operadores macro.
    pub operator_control: OperatorControl,
    /// Hilos para la evaluación (0 = automático).
    pub threads: usize,
    /// Configuración de la aptitud.
    pub fitness: FitnessConfig,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            population: 100,
            generations: 200,
            width: 120,
            selection: Selection::Tournament(3),
            crossover: Crossover::Columns,
            p_crossover: 0.9,
            mutations_per_child: 3.0,
            p_macro: 0.3,
            p_style: 0.05,
            elitism: 2,
            repair: true,
            operator_control: OperatorControl::Uniform,
            threads: 0,
            fitness: FitnessConfig::default(),
        }
    }
}

/// Estadísticas de una generación.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenStats {
    /// Generación (0 = población inicial).
    pub generation: usize,
    /// Evaluaciones acumuladas.
    pub evaluations: usize,
    /// Mejor aptitud de la generación.
    pub best: f64,
    /// Mejor aptitud histórica.
    pub best_ever: f64,
    /// Aptitud media.
    pub mean: f64,
    /// Peor aptitud.
    pub worst: f64,
    /// Desviación típica de la aptitud.
    pub std: f64,
    /// Fracción de individuos jugables.
    pub feasible: f64,
    /// Distancia de Hamming media entre pares, normalizada a `[0, 1]`.
    pub diversity: f64,
    /// Entropía genotípica media por celda, normalizada a `[0, 1]`.
    pub cell_entropy: f64,
    /// Entropía del gen de estilo, normalizada a `[0, 1]`.
    pub style_entropy: f64,
    /// Individuos por estilo.
    pub style_counts: [usize; Style::COUNT],
    /// Intensidad de selección realizada: (media de los padres − media)/σ.
    pub selection_intensity: f64,
    /// Calidad, dificultad, variedad y coherencia del mejor de la generación.
    pub best_objectives: [f64; 4],
    /// Uso de cada operador macro (bandido) en la generación.
    pub macro_usage: [f64; MacroOp::ALL.len()],
}

/// Resultado de una ejecución.
#[derive(Debug, Clone)]
pub struct RunResult {
    /// Historial por generación.
    pub history: Vec<GenStats>,
    /// Mejor nivel encontrado.
    pub best: Level,
    /// Evaluación del mejor nivel.
    pub best_eval: Evaluation,
    /// Mejor nivel jugable por estilo (salón de la fama).
    pub best_by_style: Vec<Option<(Level, Evaluation)>>,
    /// Población final con sus evaluaciones.
    pub final_population: Vec<(Level, Evaluation)>,
    /// Segundos de ejecución.
    pub seconds: f64,
}

/// Número de hilos efectivo (se consulta una sola vez por ejecución).
pub fn effective_threads(requested: usize) -> usize {
    if requested > 0 {
        requested
    } else {
        std::thread::available_parallelism().map_or(1, std::num::NonZero::get)
    }
}

/// Evalúa `levels` en paralelo escribiendo en `out`.
pub fn evaluate_all(
    levels: &[Level],
    out: &mut [Evaluation],
    evaluators: &mut [Evaluator<'_>],
    scorer: Option<&dyn Scorer>,
) {
    debug_assert_eq!(levels.len(), out.len());
    let per = levels.len().div_ceil(evaluators.len()).max(1);
    if evaluators.len() == 1 || levels.len() < 8 {
        for (l, o) in levels.iter().zip(out.iter_mut()) {
            *o = evaluators[0].evaluate(l, scorer);
        }
        return;
    }
    std::thread::scope(|s| {
        for ((ls, os), ev) in levels
            .chunks(per)
            .zip(out.chunks_mut(per))
            .zip(evaluators.iter_mut())
        {
            s.spawn(move || {
                for (l, o) in ls.iter().zip(os.iter_mut()) {
                    *o = ev.evaluate(l, scorer);
                }
            });
        }
    });
}

/// Diversidad exacta en `O(N·L)`: para cada celda, el número de pares distintos
/// es `(N² − Σ_k c_k²)/2`, con `c_k` el número de individuos con el tile `k`.
fn diversity(pop: &[Level]) -> (f64, f64) {
    let n = pop.len();
    if n < 2 {
        return (0.0, 0.0);
    }
    let w = pop[0].width();
    let cells = (w - 2 * SAFE_COLS) * HEIGHT;
    let mut diff_pairs = 0.0;
    let mut entropy = 0.0;
    let mut counts = [0_u32; Tile::COUNT];
    for i in SAFE_COLS * HEIGHT..(w - SAFE_COLS) * HEIGHT {
        counts.fill(0);
        for l in pop {
            counts[l.tiles()[i] as usize] += 1;
        }
        let sq: f64 = counts.iter().map(|&c| f64::from(c) * f64::from(c)).sum();
        diff_pairs += (n as f64 * n as f64 - sq) / 2.0;
        entropy -= counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = f64::from(c) / n as f64;
                p * p.ln()
            })
            .sum::<f64>();
    }
    let pairs = (n * (n - 1) / 2) as f64;
    let max_h = (Tile::EVOLVABLE.len() as f64).ln();
    (
        diff_pairs / pairs / cells as f64,
        entropy / cells as f64 / max_h,
    )
}

struct Engine<'a> {
    cfg: &'a GaConfig,
    rng: Rng,
    pop: Vec<Level>,
    evals: Vec<Evaluation>,
    next: Vec<Level>,
    next_evals: Vec<Evaluation>,
    parents: Vec<u32>,
    scaled: Vec<f64>,
    prefix: Vec<f64>,
    macro_arm: Vec<Option<usize>>,
    parent_fit: Vec<f64>,
    bandit: Dmab,
}

impl Engine<'_> {
    fn better(&self, a: usize, b: usize) -> bool {
        if self.cfg.fitness.aggregation == Aggregation::Deb {
            self.evals[a].deb_beats(&self.evals[b])
        } else {
            self.evals[a].fitness > self.evals[b].fitness
        }
    }

    fn select(&mut self, n: usize) {
        self.parents.clear();
        let fitness: Vec<f64> = self.evals.iter().map(|e| e.fitness).collect();
        match self.cfg.selection {
            Selection::Tournament(k) => {
                let mut rng = self.rng.clone();
                let mut out = std::mem::take(&mut self.parents);
                selection::tournament(
                    self.pop.len(),
                    k,
                    n,
                    &mut rng,
                    |a, b| self.better(a, b),
                    &mut out,
                );
                self.parents = out;
                self.rng = rng;
            }
            Selection::Roulette { scaling, sus } => {
                scale(&fitness, scaling, &mut self.scaled);
                if sus {
                    selection::sus(&self.scaled, n, &mut self.rng, &mut self.parents);
                    self.rng.shuffle(&mut self.parents);
                } else {
                    selection::roulette(
                        &self.scaled,
                        n,
                        &mut self.rng,
                        &mut self.prefix,
                        &mut self.parents,
                    );
                }
            }
        }
    }

    /// Crea los hijos en `next[elites..]`.
    fn breed(&mut self, elites: usize) {
        let n = self.pop.len();
        let cells = ((self.cfg.width - 2 * SAFE_COLS) * HEIGHT) as f64;
        let mutation = Mutation {
            per_tile: self.cfg.mutations_per_child / cells,
            macro_rate: self.cfg.p_macro,
            style_rate: self.cfg.p_style,
        };
        let mut i = elites;
        let mut k = 0;
        while i < n {
            let (pa, pb) = (
                self.parents[k % self.parents.len()] as usize,
                self.parents[(k + 1) % self.parents.len()] as usize,
            );
            k += 2;
            let (a, b) = (&self.pop[pa], &self.pop[pb]);
            let pf = self.evals[pa].fitness.max(self.evals[pb].fitness);
            // Los dos hijos se escriben en next[i] y next[i+1] (si cabe).
            let (left, right) = self.next.split_at_mut(i + 1);
            let ca = &mut left[i];
            ca.clone_from(a);
            let mut spare = None;
            let cb: &mut Level = if let Some(slot) = right.first_mut() {
                slot.clone_from(b);
                slot
            } else {
                spare.insert(b.clone())
            };
            if self.rng.chance(self.cfg.p_crossover) {
                variation::crossover(self.cfg.crossover, a, b, ca, cb, &mut self.rng);
            }
            let produced = if i + 1 < n { 2 } else { 1 };
            for (j, child) in [ca, cb].into_iter().enumerate().take(produced) {
                variation::point_mutation(child, mutation.per_tile, &mut self.rng);
                let mut arm = None;
                if self.rng.chance(mutation.macro_rate) {
                    let op = match self.cfg.operator_control {
                        OperatorControl::Uniform => self.rng.index(MacroOp::ALL.len()),
                        OperatorControl::Bandit => self.bandit.choose(),
                    };
                    variation::macro_mutation(child, MacroOp::ALL[op], &mut self.rng);
                    arm = Some(op);
                }
                if self.rng.chance(mutation.style_rate) {
                    variation::style_mutation(child, &mut self.rng);
                }
                init::repair(child, self.cfg.repair);
                self.macro_arm[i + j] = arm;
                self.parent_fit[i + j] = pf;
            }
            i += produced;
        }
    }

    fn stats(&self, generation: usize, evaluations: usize, best_ever: f64) -> GenStats {
        let n = self.evals.len() as f64;
        let fit: Vec<f64> = self.evals.iter().map(|e| e.fitness).collect();
        let mean = fit.iter().sum::<f64>() / n;
        let std = (fit.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / n).sqrt();
        let (bi, best) = fit
            .iter()
            .copied()
            .enumerate()
            .fold(
                (0, f64::MIN),
                |acc, (i, f)| if f > acc.1 { (i, f) } else { acc },
            );
        let mut style_counts = [0; Style::COUNT];
        for l in &self.pop {
            style_counts[l.style.index()] += 1;
        }
        let style_entropy = -style_counts
            .iter()
            .filter(|&&c| c > 0)
            .map(|&c| {
                let p = c as f64 / n;
                p * p.ln()
            })
            .sum::<f64>()
            / (Style::COUNT as f64).ln();
        let (div, ent) = diversity(&self.pop);
        let parents_mean = if self.parents.is_empty() {
            mean
        } else {
            self.parents.iter().map(|&p| fit[p as usize]).sum::<f64>() / self.parents.len() as f64
        };
        let e = &self.evals[bi];
        let mut usage = [0.0; MacroOp::ALL.len()];
        if self.cfg.operator_control == OperatorControl::Bandit {
            usage.copy_from_slice(&self.bandit.usage());
        }
        GenStats {
            generation,
            evaluations,
            best,
            best_ever: best_ever.max(best),
            mean,
            worst: fit.iter().copied().fold(f64::MAX, f64::min),
            std,
            feasible: self.evals.iter().filter(|e| e.feasible).count() as f64 / n,
            diversity: div,
            cell_entropy: ent,
            style_entropy,
            style_counts,
            selection_intensity: if std > 0.0 {
                (parents_mean - mean) / std
            } else {
                0.0
            },
            best_objectives: [e.quality, e.difficulty, e.variety, e.coherence],
            macro_usage: usage,
        }
    }
}

/// Ejecuta el algoritmo genético. `observer` recibe las estadísticas de cada
/// generación y la población actual (para visualizar en vivo).
///
/// # Panics
///
/// Si `population < 2` o `elitism >= population`.
pub fn run(
    cfg: &GaConfig,
    checker: &Checker,
    scorer: Option<&dyn Scorer>,
    mut observer: impl FnMut(&GenStats, &[Level], &[Evaluation]),
) -> RunResult {
    assert!(
        cfg.population >= 2 && cfg.elitism < cfg.population,
        "población o elitismo inválidos"
    );
    let t0 = Instant::now();
    let n = cfg.population;
    let threads = effective_threads(cfg.threads).min(n);
    let mut evaluators: Vec<Evaluator<'_>> = (0..threads)
        .map(|_| Evaluator::new(checker, &cfg.fitness))
        .collect();
    let mut rng = Rng::seed_from_u64(cfg.seed);
    let pop: Vec<Level> = (0..n)
        .map(|_| init::random_level(cfg.width, &mut rng))
        .collect();
    let mut evals = vec![Evaluation::default(); n];
    evaluate_all(&pop, &mut evals, &mut evaluators, scorer);
    let mut e = Engine {
        cfg,
        rng,
        next: pop.clone(),
        next_evals: evals.clone(),
        pop,
        evals,
        parents: Vec::with_capacity(n),
        scaled: Vec::with_capacity(n),
        prefix: Vec::with_capacity(n),
        macro_arm: vec![None; n],
        parent_fit: vec![0.0; n],
        bandit: Dmab::new(MacroOp::ALL.len()),
    };
    let mut evaluations = n;
    let mut history = Vec::with_capacity(cfg.generations + 1);
    let mut best_by_style: Vec<Option<(Level, Evaluation)>> = vec![None; Style::COUNT];
    let mut order: Vec<usize> = (0..n).collect();
    let track =
        |pop: &[Level], evals: &[Evaluation], hof: &mut Vec<Option<(Level, Evaluation)>>| {
            for (l, ev) in pop.iter().zip(evals) {
                let slot = &mut hof[l.style.index()];
                if ev.feasible && slot.as_ref().is_none_or(|(_, b)| ev.fitness > b.fitness) {
                    *slot = Some((l.clone(), ev.clone()));
                }
            }
        };
    track(&e.pop, &e.evals, &mut best_by_style);
    let mut best_ever = f64::MIN;
    let s0 = e.stats(0, evaluations, best_ever);
    best_ever = s0.best_ever;
    observer(&s0, &e.pop, &e.evals);
    history.push(s0);

    for generation in 1..=cfg.generations {
        // 1. Elitismo: los E mejores pasan intactos (con su evaluación).
        order.sort_by(|&a, &b| {
            e.evals[b]
                .fitness
                .total_cmp(&e.evals[a].fitness)
                .then(a.cmp(&b))
        });
        for (slot, &src) in order.iter().take(cfg.elitism).enumerate() {
            e.next[slot].clone_from(&e.pop[src]);
            e.next_evals[slot].clone_from(&e.evals[src]);
        }
        // 2-3. Selección y variación.
        e.select(n - cfg.elitism + 1);
        e.breed(cfg.elitism);
        // 4. Evaluación paralela de los hijos.
        let (_, children) = e.next.split_at(cfg.elitism);
        let (_, child_evals) = e.next_evals.split_at_mut(cfg.elitism);
        evaluate_all(children, child_evals, &mut evaluators, scorer);
        evaluations += n - cfg.elitism;
        // Crédito para el bandido: mejora relativa del hijo sobre el mejor padre.
        if cfg.operator_control == OperatorControl::Bandit {
            for i in cfg.elitism..n {
                if let Some(arm) = e.macro_arm[i] {
                    let gain = (e.next_evals[i].fitness - e.parent_fit[i]).max(0.0);
                    e.bandit.update(arm, (gain / 0.05).min(1.0));
                }
            }
        }
        // 5. Intercambio de búferes.
        std::mem::swap(&mut e.pop, &mut e.next);
        std::mem::swap(&mut e.evals, &mut e.next_evals);
        track(&e.pop, &e.evals, &mut best_by_style);
        let st = e.stats(generation, evaluations, best_ever);
        best_ever = st.best_ever;
        observer(&st, &e.pop, &e.evals);
        history.push(st);
    }

    let bi = (0..n)
        .max_by(|&a, &b| e.evals[a].fitness.total_cmp(&e.evals[b].fitness))
        .unwrap_or(0);
    RunResult {
        history,
        best: e.pop[bi].clone(),
        best_eval: e.evals[bi].clone(),
        best_by_style,
        final_population: e.pop.into_iter().zip(e.evals).collect(),
        seconds: t0.elapsed().as_secs_f64(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small(seed: u64, threads: usize) -> GaConfig {
        GaConfig {
            seed,
            population: 24,
            generations: 8,
            width: 60,
            threads,
            ..GaConfig::default()
        }
    }

    #[test]
    fn determinista_con_cualquier_numero_de_hilos() {
        let checker = Checker::default();
        let a = run(&small(7, 1), &checker, None, |_, _, _| {});
        let b = run(&small(7, 4), &checker, None, |_, _, _| {});
        assert_eq!(a.history, b.history);
        assert_eq!(a.best, b.best);
    }

    #[test]
    fn elitismo_hace_monotono_el_mejor() {
        let checker = Checker::default();
        let r = run(&small(3, 2), &checker, None, |_, _, _| {});
        for w in r.history.windows(2) {
            assert!(w[1].best >= w[0].best - 1e-12);
        }
        assert_eq!(r.history.last().map(|s| s.evaluations), Some(24 + 8 * 22));
    }

    #[test]
    fn todas_las_selecciones_funcionan() {
        let checker = Checker::default();
        for sel in [
            Selection::Tournament(2),
            Selection::Roulette {
                scaling: selection::Scaling::None,
                sus: false,
            },
            Selection::Roulette {
                scaling: selection::Scaling::Linear(1.5),
                sus: true,
            },
            Selection::Roulette {
                scaling: selection::Scaling::Sigma(2.0),
                sus: true,
            },
        ] {
            let cfg = GaConfig {
                selection: sel,
                operator_control: OperatorControl::Bandit,
                ..small(1, 2)
            };
            let r = run(&cfg, &checker, None, |_, _, _| {});
            assert!(r.best_eval.fitness > 0.0);
        }
    }
}
