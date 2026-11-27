//! Ejecutor de experimentos con números aleatorios comunes (las semillas
//! `0..R` son las mismas para todas las variantes) y análisis estadístico.
//!
//! Cada experimento es una lista de variantes (algoritmo + configuración). Las
//! ejecuciones se reparten entre hilos con `std::thread::scope`; cada ejecución
//! usa un solo hilo, así el resultado no depende del número de hilos.

use crate::baselines::{one_plus_one, random_search};
use crate::error::{Error, Result};
use crate::fitness::{Aggregation, Evaluation, Extras, Scorer};
use crate::ga::selection::{Scaling, Selection};
use crate::ga::variation::Crossover;
use crate::ga::{self, GaConfig, Niching, OperatorControl};
use crate::level::{Level, Style};
use crate::play::Checker;
use crate::plot::{self, Series};
use crate::stats;
use std::fmt::Write as _;
use std::path::Path;

/// Algoritmo de una variante.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// Algoritmo genético.
    Ga,
    /// Búsqueda aleatoria.
    Random,
    /// (1+1)-EA.
    OnePlusOne,
}

/// Variante de un experimento.
#[derive(Debug, Clone)]
pub struct Variant {
    /// Nombre corto.
    pub name: String,
    /// Instancia del problema (para Friedman); vacía si solo hay una.
    pub instance: String,
    /// Algoritmo.
    pub algorithm: Algorithm,
    /// Configuración (la semilla se sustituye en cada ejecución).
    pub config: GaConfig,
}

/// Resumen de una ejecución.
#[derive(Debug, Clone)]
pub struct RunSummary {
    /// Índice de la variante.
    pub variant: usize,
    /// Semilla.
    pub seed: u64,
    /// Mejor aptitud final.
    pub best: f64,
    /// Evaluación del mejor.
    pub eval: Evaluation,
    /// Mejor nivel.
    pub level: Level,
    /// Evaluaciones hasta el primer jugable.
    pub first_feasible: Option<usize>,
    /// Mejor histórico por generación.
    pub curve: Vec<f64>,
    /// Evaluaciones acumuladas por generación.
    pub evaluations: Vec<usize>,
    /// Diversidad final (solo GA; `NaN` en otro caso).
    pub final_diversity: f64,
    /// Intensidad de selección media realizada (solo GA).
    pub mean_intensity: f64,
    /// Segundos.
    pub seconds: f64,
}

/// Ejecuta una variante con una semilla.
pub fn run_variant(
    index: usize,
    v: &Variant,
    seed: u64,
    checker: &Checker,
    extras: Extras<'_>,
) -> RunSummary {
    let cfg = GaConfig {
        seed,
        threads: 1,
        ..v.config.clone()
    };
    let mut r = run_variant_raw(index, v, &cfg, checker, extras);
    // Misma resolución para todos los algoritmos: el primer lote cuya mejor
    // aptitud indica un nivel jugable (los jugables valen al menos 0,1).
    r.first_feasible = r
        .curve
        .iter()
        .zip(&r.evaluations)
        .find(|(f, _)| **f >= 0.1)
        .map(|(_, &e)| e);
    r
}

fn run_variant_raw(
    index: usize,
    v: &Variant,
    cfg: &GaConfig,
    checker: &Checker,
    extras: Extras<'_>,
) -> RunSummary {
    let seed = cfg.seed;
    match v.algorithm {
        Algorithm::Ga => {
            let r = ga::run(cfg, checker, extras, |_, _, _| {});
            let first_feasible = r
                .history
                .iter()
                .find(|h| h.feasible > 0.0)
                .map(|h| h.evaluations);
            let n = r.history.len().max(2) as f64 - 1.0;
            RunSummary {
                variant: index,
                seed,
                best: r.best_eval.fitness,
                eval: r.best_eval,
                level: r.best,
                first_feasible,
                curve: r.history.iter().map(|h| h.best_ever).collect(),
                evaluations: r.history.iter().map(|h| h.evaluations).collect(),
                final_diversity: r.history.last().map_or(f64::NAN, |h| h.diversity),
                mean_intensity: r
                    .history
                    .iter()
                    .skip(1)
                    .map(|h| h.selection_intensity)
                    .sum::<f64>()
                    / n,
                seconds: r.seconds,
            }
        }
        Algorithm::Random | Algorithm::OnePlusOne => {
            let r = if v.algorithm == Algorithm::Random {
                random_search(cfg, checker, extras)
            } else {
                one_plus_one(cfg, checker, extras)
            };
            RunSummary {
                variant: index,
                seed,
                best: r.best_eval.fitness,
                eval: r.best_eval,
                level: r.best,
                first_feasible: r.first_feasible,
                curve: r.curve,
                evaluations: r.evaluations,
                final_diversity: f64::NAN,
                mean_intensity: f64::NAN,
                seconds: r.seconds,
            }
        }
    }
}

/// Ejecuta todas las variantes con las semillas `0..runs` en paralelo.
/// `progress` recibe el número de ejecuciones terminadas.
///
/// # Panics
///
/// Si alguna ejecución entra en pánico (se propaga al hilo principal).
pub fn run_all(
    variants: &[Variant],
    runs: u64,
    threads: usize,
    extras: Extras<'_>,
    progress: impl Fn(usize, usize) + Sync,
) -> Vec<RunSummary> {
    let jobs: Vec<(usize, u64)> = (0..variants.len())
        .flat_map(|v| (0..runs).map(move |s| (v, s)))
        .collect();
    let total = jobs.len();
    let done = std::sync::atomic::AtomicUsize::new(0);
    let threads = threads.max(1).min(total.max(1));
    let mut results: Vec<Option<RunSummary>> = vec![None; total];
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let (jobs, done, progress) = (&jobs, &done, &progress);
                s.spawn(move || {
                    let checker = Checker::default();
                    // Reparto estático por turnos: determinista e independiente del orden.
                    jobs.iter()
                        .enumerate()
                        .skip(t)
                        .step_by(threads)
                        .map(|(i, &(v, seed))| {
                            let r = run_variant(v, &variants[v], seed, &checker, extras);
                            let k = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                            progress(k, total);
                            (i, r)
                        })
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        for h in handles {
            for (i, r) in h.join().expect("un hilo de experimento entró en pánico") {
                results[i] = Some(r);
            }
        }
    });
    results.into_iter().flatten().collect()
}

fn ga_variant(name: impl Into<String>, config: GaConfig) -> Variant {
    Variant {
        name: name.into(),
        instance: String::new(),
        algorithm: Algorithm::Ga,
        config,
    }
}

/// Nombres de los experimentos disponibles.
pub const EXPERIMENTS: [(&str, &str); 11] = [
    (
        "e1",
        "¿Aporta el GA? GA frente a búsqueda aleatoria, (1+1)-EA y GA sin cruce",
    ),
    (
        "e2",
        "Selección: torneo de varios tamaños frente a ruleta con y sin escalado",
    ),
    ("e3", "Elitismo: 0, 1, 2, 5 y 10 élites"),
    ("e4", "Cruce: columnas, bloque 2D, uniforme y ninguno"),
    (
        "e5",
        "Sensibilidad: probabilidad de cruce por tasa de mutación (mapa de calor)",
    ),
    (
        "e6",
        "Agregación: suma ponderada, media geométrica, penalización y reglas de Deb",
    ),
    (
        "e7",
        "Instancias: tres dificultades objetivo por tres anchos (prueba de Friedman)",
    ),
    (
        "e8",
        "Red de naturalidad: GA con y sin el término de cross-entropy",
    ),
    ("e9", "Rango expresivo de los mejores niveles"),
    (
        "e10",
        "Control de operadores: uniforme frente a bandido D-MAB",
    ),
    (
        "e11",
        "Diversidad: sin nichos, fitness sharing y crowding determinista",
    ),
];

/// Define las variantes de un experimento a partir de una configuración base.
///
/// # Errors
///
/// Si el nombre no corresponde a ningún experimento.
pub fn define(name: &str, base: &GaConfig) -> Result<Vec<Variant>> {
    let b = || base.clone();
    Ok(match name {
        "e1" => vec![
            ga_variant("GA", b()),
            Variant {
                name: "aleatoria".into(),
                instance: String::new(),
                algorithm: Algorithm::Random,
                config: b(),
            },
            Variant {
                name: "(1+1)-EA".into(),
                instance: String::new(),
                algorithm: Algorithm::OnePlusOne,
                config: b(),
            },
            ga_variant(
                "GA sin cruce",
                GaConfig {
                    crossover: Crossover::None,
                    ..b()
                },
            ),
        ],
        "e2" => define_e2(&b),
        "e3" => [2, 0, 1, 5, 10]
            .iter()
            .map(|&e| ga_variant(format!("E={e}"), GaConfig { elitism: e, ..b() }))
            .collect(),
        "e4" => [
            Crossover::Columns,
            Crossover::Block,
            Crossover::Uniform,
            Crossover::None,
        ]
        .iter()
        .map(|&c| {
            ga_variant(
                c.name(),
                GaConfig {
                    crossover: c,
                    ..b()
                },
            )
        })
        .collect(),
        "e5" => {
            let mut v = Vec::new();
            for pc in [0.6, 0.8, 0.9, 1.0] {
                for k in [0.5, 1.0, 3.0, 5.0] {
                    v.push(ga_variant(
                        format!("pc={pc} k={k}"),
                        GaConfig {
                            p_crossover: pc,
                            mutations_per_child: k,
                            ..b()
                        },
                    ));
                }
            }
            v
        }
        "e6" => [
            ("suma", Aggregation::WeightedSum),
            ("geométrica", Aggregation::Geometric),
            ("penalización λ=1", Aggregation::Penalty(1.0)),
            ("Deb", Aggregation::Deb),
        ]
        .iter()
        .map(|&(n, a)| {
            let mut c = b();
            c.fitness.aggregation = a;
            ga_variant(n, c)
        })
        .collect(),
        "e7" => define_e7(&b),
        "e8" => define_e8(&b),
        "e9" => vec![ga_variant("GA", b())],
        "e10" => vec![
            ga_variant("uniforme", b()),
            ga_variant(
                "bandido D-MAB",
                GaConfig {
                    operator_control: OperatorControl::Bandit,
                    ..b()
                },
            ),
        ],
        "e11" => define_e11(&b),
        _ => {
            return Err(Error::Config(format!(
                "experimento desconocido «{name}»; disponibles: {}",
                EXPERIMENTS.map(|e| e.0).join(", ")
            )));
        }
    })
}

fn define_e2(b: &dyn Fn() -> GaConfig) -> Vec<Variant> {
    let mut v: Vec<Variant> = [2, 3, 5, 7]
        .iter()
        .map(|&k| {
            ga_variant(
                format!("torneo-{k}"),
                GaConfig {
                    selection: Selection::Tournament(k),
                    ..b()
                },
            )
        })
        .collect();
    for (n, scaling, sus) in [
        ("ruleta cruda", Scaling::None, false),
        ("ruleta lineal", Scaling::Linear(1.5), false),
        ("ruleta lineal SUS", Scaling::Linear(1.5), true),
        ("ruleta sigma SUS", Scaling::Sigma(2.0), true),
    ] {
        v.push(ga_variant(
            n,
            GaConfig {
                selection: Selection::Roulette { scaling, sus },
                ..b()
            },
        ));
    }
    v
}

fn define_e7(b: &dyn Fn() -> GaConfig) -> Vec<Variant> {
    let mut v = Vec::new();
    for d in [0.3, 0.5, 0.7] {
        for w in [80, 120, 160] {
            let mut c = GaConfig { width: w, ..b() };
            c.fitness.targets.difficulty = d;
            let inst = format!("D*={d} W={w}");
            for (n, alg, sel) in [
                ("torneo-3", Algorithm::Ga, Selection::Tournament(3)),
                (
                    "ruleta lineal SUS",
                    Algorithm::Ga,
                    Selection::Roulette {
                        scaling: Scaling::Linear(1.5),
                        sus: true,
                    },
                ),
                ("(1+1)-EA", Algorithm::OnePlusOne, Selection::Tournament(3)),
                ("aleatoria", Algorithm::Random, Selection::Tournament(3)),
            ] {
                v.push(Variant {
                    name: n.into(),
                    instance: inst.clone(),
                    algorithm: alg,
                    config: GaConfig {
                        selection: sel,
                        ..c.clone()
                    },
                });
            }
        }
    }
    v
}

fn define_e8(b: &dyn Fn() -> GaConfig) -> Vec<Variant> {
    let mut with = b();
    with.fitness.weights.naturalness = 0.15;
    let mut without = b();
    without.fitness.weights.naturalness = 0.0;
    vec![
        ga_variant("GA", without),
        ga_variant("GA + naturalidad", with),
    ]
}

fn define_e11(b: &dyn Fn() -> GaConfig) -> Vec<Variant> {
    [
        ("sin nichos", Niching::None),
        ("sharing σ=0,1", Niching::Sharing(0.1)),
        ("sharing σ=0,2", Niching::Sharing(0.2)),
        ("crowding (Mahfoud)", Niching::Crowding),
        ("crowding + torneo", Niching::CrowdingSelective),
    ]
    .iter()
    .map(|&(n, niching)| ga_variant(n, GaConfig { niching, ..b() }))
    .collect()
}

/// Umbral de éxito: jugable y con aptitud al menos este valor.
pub const SUCCESS: f64 = 0.9;

fn success(r: &RunSummary) -> bool {
    r.eval.feasible && r.best >= SUCCESS
}

/// Mediana y cuartiles de la curva de cada variante, generación a generación.
fn curve_series(name: &str, runs: &[&RunSummary]) -> Series {
    let len = runs.iter().map(|r| r.curve.len()).min().unwrap_or(0);
    let mut points = Vec::with_capacity(len);
    let mut band = Vec::with_capacity(len);
    for g in 0..len {
        let mut v: Vec<f64> = runs.iter().map(|r| r.curve[g]).collect();
        v.sort_by(f64::total_cmp);
        let x = runs[0].evaluations[g] as f64;
        points.push((x, plot::quantile(&v, 0.5)));
        band.push((x, plot::quantile(&v, 0.25), plot::quantile(&v, 0.75)));
    }
    Series {
        name: name.into(),
        points,
        band,
    }
}

/// Cabecera del CSV por ejecución.
const RUNS_HEADER: &str = "variante,instancia,semilla,mejor,jugable,exito,calidad,dificultad,variedad,coherencia,estilo,primer_jugable,diversidad_final,intensidad_media,segundos\n";

/// Resultados agrupados por nombre de variante (todas las instancias juntas).
struct Analysis<'a> {
    variants: &'a [Variant],
    runs: &'a [RunSummary],
    names: Vec<String>,
}

impl<'a> Analysis<'a> {
    fn new(variants: &'a [Variant], runs: &'a [RunSummary]) -> Self {
        let mut names: Vec<String> = Vec::new();
        for v in variants {
            if !names.contains(&v.name) {
                names.push(v.name.clone());
            }
        }
        Self {
            variants,
            runs,
            names,
        }
    }

    fn by_name(&self, n: &str) -> Vec<&'a RunSummary> {
        self.runs
            .iter()
            .filter(|r| self.variants[r.variant].name == n)
            .collect()
    }

    fn successes(&self, n: &str) -> (u64, u64) {
        let rs = self.by_name(n);
        (
            rs.iter().filter(|r| success(r)).count() as u64,
            rs.len() as u64,
        )
    }

    /// CSV con una fila por ejecución.
    fn runs_csv(&self) -> String {
        let mut csv = String::from(RUNS_HEADER);
        for r in self.runs {
            let v = &self.variants[r.variant];
            let _ = writeln!(
                csv,
                "\"{}\",\"{}\",{},{:.6},{},{},{:.6},{:.6},{:.6},{:.6},{},{},{:.6},{:.6},{:.3}",
                v.name,
                v.instance,
                r.seed,
                r.best,
                r.eval.feasible,
                success(r),
                r.eval.quality,
                r.eval.difficulty,
                r.eval.variety,
                r.eval.coherence,
                r.level.style.name(),
                r.first_feasible.map_or(String::new(), |e| e.to_string()),
                r.final_diversity,
                r.mean_intensity,
                r.seconds
            );
        }
        csv
    }

    /// Tabla descriptiva por variante; devuelve el informe, el CSV de resumen y
    /// la aptitud final de cada grupo.
    fn summary(&self, rep: &mut String) -> (String, Vec<Vec<f64>>) {
        rep.push_str("variante | n | mediana [Q1; Q3] | media ± desv | éxito | primer jugable (mediana) | diversidad final | s/ejec.\n");
        let mut csv = String::from(
            "variante,n,mediana,q1,q3,media,desv,exitos,mediana_primer_jugable,diversidad_final,segundos\n",
        );
        let mut groups = Vec::new();
        for n in &self.names {
            let rs = self.by_name(n);
            let best: Vec<f64> = rs.iter().map(|r| r.best).collect();
            let s = stats::summary(&best);
            let (succ, _) = self.successes(n);
            let ff: Vec<f64> = rs
                .iter()
                .filter_map(|r| r.first_feasible.map(|e| e as f64))
                .collect();
            let ffm = stats::summary(&ff).median;
            let div = rs.iter().map(|r| r.final_diversity).sum::<f64>() / rs.len() as f64;
            let secs = rs.iter().map(|r| r.seconds).sum::<f64>() / rs.len() as f64;
            let _ = writeln!(
                rep,
                "{n} | {} | {:.4} [{:.4}; {:.4}] | {:.4} ± {:.4} | {succ}/{} | {ffm:.0} | {div:.3} | {secs:.2}",
                s.n, s.median, s.q1, s.q3, s.mean, s.std, s.n
            );
            let _ = writeln!(
                csv,
                "\"{n}\",{},{:.6},{:.6},{:.6},{:.6},{:.6},{succ},{ffm:.1},{div:.6},{secs:.3}",
                s.n, s.median, s.q1, s.q3, s.mean, s.std
            );
            groups.push(best);
        }
        (csv, groups)
    }

    /// Kruskal-Wallis y comparaciones con la primera variante (referencia).
    fn pairwise(&self, groups: &[Vec<f64>], rep: &mut String) {
        let (h, p) = stats::kruskal_wallis(groups);
        let _ = writeln!(rep, "\nKruskal-Wallis: H = {h:.3}, p = {p:.3e}");
        let (ref_succ, ref_n) = self.successes(&self.names[0]);
        let mut lines = Vec::new();
        let mut ps = Vec::new();
        for (i, g) in groups.iter().enumerate().skip(1) {
            let mw = stats::mann_whitney(&groups[0], g);
            let a = stats::a12(&groups[0], g);
            let (s, n) = self.successes(&self.names[i]);
            let (pf, or) = stats::fisher_exact(ref_succ, ref_n - ref_succ, s, n - s);
            ps.push(mw.p);
            lines.push((i, mw, a, pf, or));
        }
        let adj = stats::holm(&ps);
        let _ = writeln!(
            rep,
            "\nReferencia: {}. Mann-Whitney (p ajustado por Holm), Â12 y Fisher sobre el éxito:",
            self.names[0]
        );
        for ((i, mw, a, pf, or), pa) in lines.into_iter().zip(adj) {
            let _ = writeln!(
                rep,
                "  vs {}: U = {:.1}, p = {:.3e} (Holm {:.3e}); Â12 = {:.3} ({}); Fisher p = {:.3e}, OR = {:.2}",
                self.names[i],
                mw.u,
                mw.p,
                pa,
                a,
                stats::a12_magnitude(a),
                pf,
                or
            );
        }
    }

    /// Instancias distintas (para Friedman).
    fn instances(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for v in self.variants {
            if !v.instance.is_empty() && !out.contains(&v.instance) {
                out.push(v.instance.clone());
            }
        }
        out
    }

    /// Prueba de Friedman sobre las medianas por instancia.
    fn friedman(&self, instances: &[String], rep: &mut String) {
        let median_of = |n: &str, inst: &str| {
            let v: Vec<f64> = self
                .runs
                .iter()
                .filter(|r| {
                    self.variants[r.variant].name == n && self.variants[r.variant].instance == inst
                })
                .map(|r| r.best)
                .collect();
            stats::summary(&v).median
        };
        let data: Vec<Vec<f64>> = instances
            .iter()
            .map(|inst| self.names.iter().map(|n| median_of(n, inst)).collect())
            .collect();
        let (chi2, p, ranks) = stats::friedman(&data);
        let _ = writeln!(
            rep,
            "\nFriedman sobre {} instancias: χ² = {chi2:.3}, p = {p:.3e}",
            instances.len()
        );
        for (n, r) in self.names.iter().zip(ranks) {
            let _ = writeln!(rep, "  rango medio {n}: {r:.2}");
        }
    }
}

/// Analiza los resultados y escribe CSV, gráficas e informe de pruebas en `dir`.
/// Devuelve el informe de texto.
///
/// # Errors
///
/// Si no se pueden escribir los archivos.
pub fn analyze(
    dir: &Path,
    title: &str,
    variants: &[Variant],
    runs: &[RunSummary],
) -> Result<String> {
    std::fs::create_dir_all(dir)?;
    let a = Analysis::new(variants, runs);
    std::fs::write(dir.join("resultados.csv"), a.runs_csv())?;
    let mut rep = format!(
        "# {title}\n\nEjecuciones: {}. Éxito: jugable y F ≥ {SUCCESS}.\n\n",
        runs.len()
    );
    let (resumen, groups) = a.summary(&mut rep);
    std::fs::write(dir.join("resumen.csv"), resumen)?;
    if groups.len() > 1 {
        a.pairwise(&groups, &mut rep);
    }
    let instances = a.instances();
    if instances.len() > 1 {
        a.friedman(&instances, &mut rep);
    } else {
        let series: Vec<Series> = a
            .names
            .iter()
            .map(|n| curve_series(n, &a.by_name(n)))
            .collect();
        let chart = plot::line_chart(
            &format!("{title}: mejor histórico (mediana e IQR)"),
            "evaluaciones",
            "aptitud",
            &series,
        );
        std::fs::write(dir.join("convergencia.svg"), chart)?;
    }
    let boxes: Vec<(String, Vec<f64>)> = a.names.iter().cloned().zip(groups).collect();
    std::fs::write(
        dir.join("cajas.svg"),
        plot::box_plot(&format!("{title}: aptitud final"), "aptitud", &boxes),
    )?;
    std::fs::write(dir.join("informe.txt"), &rep)?;
    Ok(rep)
}

/// Mapa de calor de la mediana de la aptitud del experimento e5.
///
/// # Errors
///
/// Si no se puede escribir el archivo.
pub fn e5_heatmap(dir: &Path, variants: &[Variant], runs: &[RunSummary]) -> Result<()> {
    let pcs = [0.6, 0.8, 0.9, 1.0];
    let ks = [0.5, 1.0, 3.0, 5.0];
    let values: Vec<Vec<f64>> = pcs
        .iter()
        .map(|&pc| {
            ks.iter()
                .map(|&k| {
                    let v: Vec<f64> = runs
                        .iter()
                        .filter(|r| {
                            let c = &variants[r.variant].config;
                            (c.p_crossover - pc).abs() < 1e-9
                                && (c.mutations_per_child - k).abs() < 1e-9
                        })
                        .map(|r| r.best)
                        .collect();
                    stats::summary(&v).median
                })
                .collect()
        })
        .collect();
    let svg = plot::heatmap(
        "Mediana de la aptitud final",
        &pcs.map(|p| format!("pc={p}")),
        &ks.map(|k| format!("{k}/L")),
        &values,
        ("mutaciones esperadas por hijo", "probabilidad de cruce"),
    );
    std::fs::write(dir.join("sensibilidad.svg"), svg)?;
    Ok(())
}

/// Histogramas 2D del rango expresivo (dificultad × linealidad y variedad ×
/// coherencia) y reparto de estilos de los mejores niveles.
///
/// # Errors
///
/// Si no se puede escribir el archivo.
pub fn e9_expressive_range(dir: &Path, runs: &[RunSummary]) -> Result<String> {
    const BINS: usize = 10;
    let hist = |fx: &dyn Fn(&RunSummary) -> f64, fy: &dyn Fn(&RunSummary) -> f64| {
        let mut m = vec![vec![0.0; BINS]; BINS];
        for r in runs {
            let bx = ((fx(r) * BINS as f64) as usize).min(BINS - 1);
            let by = ((fy(r) * BINS as f64) as usize).min(BINS - 1);
            m[BINS - 1 - by][bx] += 1.0;
        }
        m
    };
    let labels: Vec<String> = (0..BINS)
        .map(|i| format!("{:.1}", i as f64 / BINS as f64))
        .collect();
    let rev: Vec<String> = labels.iter().rev().cloned().collect();
    let a = hist(&|r| r.eval.difficulty, &|r| r.eval.features.linearity);
    std::fs::write(
        dir.join("rango_dificultad_linealidad.svg"),
        plot::heatmap(
            "Rango expresivo: dificultad × linealidad",
            &rev,
            &labels,
            &a,
            ("dificultad D", "linealidad"),
        ),
    )?;
    let b = hist(&|r| r.eval.variety, &|r| r.eval.coherence);
    std::fs::write(
        dir.join("rango_variedad_coherencia.svg"),
        plot::heatmap(
            "Rango expresivo: variedad × coherencia",
            &rev,
            &labels,
            &b,
            ("variedad V", "coherencia C"),
        ),
    )?;
    let mut counts = [0; Style::COUNT];
    for r in runs {
        counts[r.level.style.index()] += 1;
    }
    let mut rep = String::from("\nEstilo del mejor nivel por ejecución:\n");
    for st in Style::ALL {
        let _ = writeln!(rep, "  {}: {}", st.name(), counts[st.index()]);
    }
    Ok(rep)
}

/// Compara la naturalidad (según el modelo) de los mejores niveles de cada
/// variante, también para las que no la optimizaban.
pub fn e8_naturalness(variants: &[Variant], runs: &[RunSummary], model: &dyn Scorer) -> String {
    let mut names: Vec<&str> = Vec::new();
    for v in variants {
        if !names.contains(&v.name.as_str()) {
            names.push(&v.name);
        }
    }
    let groups: Vec<Vec<f64>> = names
        .iter()
        .map(|n| {
            runs.iter()
                .filter(|r| variants[r.variant].name == *n)
                .map(|r| model.score(&r.level))
                .collect()
        })
        .collect();
    let mut rep = String::from(
        "\nNota: la aptitud F no es comparable entre estas variantes porque la función \
         objetivo cambia.\nObjetivos comunes del mejor nivel (mediana):\n",
    );
    for n in &names {
        let rs: Vec<&RunSummary> = runs
            .iter()
            .filter(|r| variants[r.variant].name == *n)
            .collect();
        let med = |f: &dyn Fn(&RunSummary) -> f64| {
            stats::summary(&rs.iter().map(|r| f(r)).collect::<Vec<_>>()).median
        };
        let _ = writeln!(
            rep,
            "  {n}: D = {:.3}, V = {:.3}, C = {:.3}, jugables {}/{}",
            med(&|r| r.eval.difficulty),
            med(&|r| r.eval.variety),
            med(&|r| r.eval.coherence),
            rs.iter().filter(|r| r.eval.feasible).count(),
            rs.len()
        );
    }
    rep.push_str("\nNaturalidad del mejor nivel (1 = tan natural como el corpus VGLC):\n");
    for (n, g) in names.iter().zip(&groups) {
        let s = stats::summary(g);
        let _ = writeln!(
            rep,
            "  {n}: mediana {:.3} [{:.3}; {:.3}]",
            s.median, s.q1, s.q3
        );
    }
    if groups.len() == 2 {
        let mw = stats::mann_whitney(&groups[1], &groups[0]);
        let a = stats::a12(&groups[1], &groups[0]);
        let _ = writeln!(
            rep,
            "  {} vs {}: p = {:.3e}, Â12 = {a:.3} ({})",
            names[1],
            names[0],
            mw.p,
            stats::a12_magnitude(a)
        );
    }
    rep
}

/// Compara la intensidad de selección medida de cada variante de E2 con la
/// teórica del torneo (Blickle y Thiele, 1995) e informa de su pérdida de
/// diversidad teórica.
pub fn e2_selection_pressure(variants: &[Variant], runs: &[RunSummary]) -> String {
    let mut rep = String::from(
        "\nIntensidad de selección: medida (mediana) frente a la teórica del torneo:\n",
    );
    for (i, v) in variants.iter().enumerate() {
        let measured: Vec<f64> = runs
            .iter()
            .filter(|r| r.variant == i)
            .map(|r| r.mean_intensity)
            .collect();
        let med = stats::summary(&measured).median;
        match v.config.selection {
            Selection::Tournament(t) => {
                let _ = writeln!(
                    rep,
                    "  {}: medida {med:.3}, teórica {:.3}, pérdida de diversidad teórica {:.3}",
                    v.name,
                    ga::selection::tournament_intensity(t),
                    ga::selection::tournament_loss_of_diversity(t)
                );
            }
            Selection::Roulette { .. } => {
                let _ = writeln!(rep, "  {}: medida {med:.3}", v.name);
            }
        }
    }
    rep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn todos_los_experimentos_se_definen() {
        let base = GaConfig::default();
        for (name, _) in EXPERIMENTS {
            assert!(!define(name, &base).expect("definido").is_empty(), "{name}");
        }
        assert!(define("zz", &base).is_err());
    }

    #[test]
    fn ejecucion_paralela_determinista() {
        let base = GaConfig {
            population: 12,
            generations: 3,
            width: 40,
            ..GaConfig::default()
        };
        let v = define("e1", &base).expect("definido");
        let a = run_all(&v, 2, 1, Extras::default(), |_, _| {});
        let b = run_all(&v, 2, 3, Extras::default(), |_, _| {});
        let fa: Vec<f64> = a.iter().map(|r| r.best).collect();
        let fb: Vec<f64> = b.iter().map(|r| r.best).collect();
        assert_eq!(fa, fb);
    }
}
