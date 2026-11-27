//! Interfaz de línea de órdenes de `nivelga`.

use nivelga::cli::{self, Args};
use nivelga::fitness::{Extras, Scorer};
use nivelga::ga;
use nivelga::level::Level;
use nivelga::naturalness::NaturalnessModel;
use nivelga::play::{Checker, PlayScratch};
use nivelga::render::Tilesets;
use nivelga::rl::{self, Agent, PlaytesterScorer};
use nivelga::{Error, Result, ansi, report};
use std::fmt::Write as _;
use std::path::PathBuf;

const AYUDA: &str = "\
nivelga: niveles de plataformas 2D evolucionados con un algoritmo genético

USO:
  nivelga evolucionar [opciones]      ejecuta el GA y guarda resultados
  nivelga ver <nivel.txt> [--columnas 60] [--retardo 60]
                                      anima en la terminal el camino óptimo
  nivelga dibujar <nivel.txt> <salida.png> [--escala 2]
  nivelga comprobar <nivel.txt>       analiza jugabilidad y métricas
  nivelga generar [--salida DIR] [--semilla N] [--ancho W] [--dificultad D]
                  [--estilo azar|libre|pradera|granja|dulce|industrial]
                                      evoluciona un nivel nuevo para el juego (JSON + PNG)
  nivelga exportar <nivel.txt> [--salida DIR]
                                      convierte un nivel en texto al formato del juego
  nivelga corpus [--datos data/vglc/smb]
                                      rasgos de niveles humanos (calibración)
  nivelga partidas [resultados/partidas.csv]
                                      compara la dificultad D con las partidas jugadas
  nivelga experimento <e1..e10|todos> [--repeticiones 30] [--salida DIR]
                                      [opciones de la configuración base]
  nivelga experimentos                lista los experimentos disponibles
  nivelga entrenar-rl [--pasos 2000000] [--entornos 16] [--ocultas 64,64]
                      [--salida modelos/agente.rla]
                                      entrena el agente A2C (cross-entropy ponderada)
  nivelga jugar <nivel.txt> [--agente modelos/agente.rla]
                                      anima en la terminal al agente jugando
  nivelga entrenar-pg [--muestras 20000] [--poblacion 500] [--generaciones 50]
                      [--parsimonia 0.001] [--salida resultados/pg]
                                      PG simbólica que predice la jugabilidad (BCE)
                                      frente a regresión logística y MLP
  nivelga entrenar-nn [--datos data/vglc/smb] [--salida modelos/naturalidad.nng]
                      [--epocas 80] [--ocultas 32] [--lr 0.001] [--suavizado 0.05]
                                      entrena la red de naturalidad (cross-entropy)

OPCIONES DE `evolucionar`:
  --semilla N          semilla (0)
  --poblacion N        tamaño de población (100)
  --generaciones N     generaciones (200)
  --ancho N            columnas del nivel (120)
  --seleccion S        torneo:K | ruleta[:cruda|:lineal:C|:sigma:C][:sus]  (torneo:3)
  --cruce C            columnas | bloque | uniforme | ninguno  (columnas)
  --pc P               probabilidad de cruce (0.9)
  --mutaciones K       cambios puntuales esperados por hijo (3)
  --macro P            probabilidad de mutación macro (0.3)
  --p-estilo P         probabilidad de mutar el tileset (0.05)
  --elitismo E         élites por generación (2)
  --reparar si|no      reparación completa (si)
  --control C          uniforme | bandido  (operadores macro)
  --nicho N            ninguno | sharing[:σ] | crowding | crowding-seleccion
  --agregacion A       suma | geometrica | penalizacion:λ | deb  (suma)
  --dificultad D       dificultad objetivo D* (0.5)
  --estilo E           fija el tileset (pradera, granja, dulce, industrial)
  --hilos N            hilos de evaluación (0 = todos)
  --naturalidad RUTA   añade el término de la red de naturalidad
  --peso-nat W         peso de ese término (0.15)
  --agente RUTA        añade la dificultad empírica del agente de RL
  --peso-agente W      peso de ese término (0.15)
  --objetivo-victorias tasa de victorias objetivo del agente (0.6)
  --salida DIR         directorio de resultados (resultados/<semilla>)
  --recursos DIR       directorio de tilesets (assets)
";

fn evolucionar(mut args: Args) -> Result<()> {
    let cfg = cli::ga_config(&mut args)?;
    let out = PathBuf::from(
        args.take("salida")
            .unwrap_or_else(|| format!("resultados/semilla_{}", cfg.seed)),
    );
    let assets = PathBuf::from(args.take("recursos").unwrap_or_else(|| "assets".into()));
    let ext = Externals::load(&mut args)?;
    let mut cfg = cfg;
    if ext.model.is_some() {
        cfg.fitness.weights.naturalness = args.get("peso-nat", 0.15)?;
    }
    if ext.agent.is_some() {
        cfg.fitness.weights.playtester = args.get("peso-agente", 0.15)?;
    }
    args.finish()?;
    let checker = Checker::default();
    let result = ga::run(&cfg, &checker, ext.extras(), |s, _, _| {
        if s.generation % 10 == 0 || s.generation == cfg.generations {
            eprintln!(
                "gen {:4} | mejor {:.4} | media {:.4} | jugables {:5.1}% | diversidad {:.3} | estilos {:?}",
                s.generation,
                s.best,
                s.mean,
                100.0 * s.feasible,
                s.diversity,
                s.style_counts
            );
        }
    });
    report::save_run(&out, &cfg, &result, &checker, &assets)?;
    println!(
        "{:.2} s · mejor nivel ({}):",
        result.seconds,
        result.best.style.name()
    );
    println!("{}", report::describe(&result.best_eval));
    println!("resultados en {}", out.display());
    Ok(())
}

/// Modelos externos opcionales: la red de naturalidad (`--naturalidad RUTA`)
/// y el agente probador de RL (`--agente RUTA`).
struct Externals {
    model: Option<NaturalnessModel>,
    agent: Option<PlaytesterScorer>,
}

impl Externals {
    fn load(args: &mut Args) -> Result<Self> {
        let model = args
            .take("naturalidad")
            .map(|p| NaturalnessModel::from_bytes(&std::fs::read(p)?))
            .transpose()?;
        let agent = match args.take("agente") {
            None => None,
            Some(p) => Some(PlaytesterScorer {
                agent: Agent::from_bytes(&std::fs::read(p)?)?,
                checker: Checker::default(),
                episodes: args.get("episodios", 4_usize)?,
                target: args.get("objetivo-victorias", 0.6)?,
                sigma: 0.2,
            }),
        };
        Ok(Self { model, agent })
    }

    fn extras(&self) -> Extras<'_> {
        Extras {
            naturalness: self.model.as_ref().map(|m| m as &dyn Scorer),
            playtester: self.agent.as_ref().map(|a| a as &dyn Scorer),
        }
    }
}

/// Guarda el historial del entrenamiento en CSV y su curva en SVG.
fn save_training(
    out: &std::path::Path,
    history: &[nivelga::naturalness::EpochLog],
    baselines: &[(&str, f64)],
    markov: f64,
) -> Result<()> {
    let report = NaturalnessModel::report(history, baselines);
    std::fs::write(out.with_extension("csv"), &report)?;
    let line =
        |name: &str, f: &dyn Fn(&nivelga::naturalness::EpochLog) -> f64| nivelga::plot::Series {
            name: name.into(),
            points: history.iter().map(|h| (h.epoch as f64, f(h))).collect(),
            band: vec![],
        };
    let series = vec![
        line("pérdida de entrenamiento", &|h| h.train_loss),
        line("NLL de validación", &|h| h.val_nll),
        line("Markov 2D (val.)", &|_| markov),
    ];
    std::fs::write(
        out.with_extension("svg"),
        nivelga::plot::line_chart(
            "Entrenamiento con cross-entropy",
            "época",
            "NLL por tile",
            &series,
        ),
    )?;
    Ok(())
}

fn entrenar_nn(mut args: Args) -> Result<()> {
    use nivelga::naturalness::{self, TrainConfig};
    let data = args.take("datos").unwrap_or_else(|| "data/vglc/smb".into());
    let out = PathBuf::from(
        args.take("salida")
            .unwrap_or_else(|| "modelos/naturalidad.nng".into()),
    );
    let d = TrainConfig::default();
    let hidden = match args.take("ocultas") {
        Some(h) => h
            .split(',')
            .map(|v| {
                v.trim()
                    .parse()
                    .map_err(|_| Error::Config(format!("capa oculta inválida «{v}»")))
            })
            .collect::<Result<Vec<usize>>>()?,
        None => d.hidden.clone(),
    };
    let cfg = TrainConfig {
        hidden,
        epochs: args.get("epocas", d.epochs)?,
        batch: args.get("lote", d.batch)?,
        lr: args.get("lr", d.lr)?,
        smoothing: args.get("suavizado", d.smoothing)?,
        patience: args.get("paciencia", d.patience)?,
        seed: args.get("semilla", d.seed)?,
        activation: match args.take("activacion").as_deref() {
            None | Some("relu") => nivelga::nn::Activation::Relu,
            Some("tanh") => nivelga::nn::Activation::Tanh,
            Some(o) => return Err(Error::Config(format!("activación desconocida «{o}»"))),
        },
    };
    args.finish()?;
    let levels = nivelga::vglc::load_dir(&data)?;
    if levels.len() < 4 {
        return Err(Error::Config(format!(
            "hacen falta al menos 4 niveles en {data}; ejecuta scripts/descargar_vglc.sh"
        )));
    }
    // Validación: uno de cada cinco niveles (determinista).
    let (val, train): (Vec<_>, Vec<_>) = levels
        .into_iter()
        .enumerate()
        .partition(|(i, _)| i % 5 == 4);
    let names: Vec<&str> = val.iter().map(|(_, (n, _))| n.as_str()).collect();
    eprintln!("validación: {names:?}");
    let train_levels: Vec<Level> = train.into_iter().map(|(_, (_, l))| l).collect();
    let val_levels: Vec<Level> = val.into_iter().map(|(_, (_, l))| l).collect();
    let (st, sv) = (
        naturalness::samples(&train_levels),
        naturalness::samples(&val_levels),
    );
    eprintln!(
        "muestras: {} de entrenamiento y {} de validación",
        st.len(),
        sv.len()
    );
    let uni = naturalness::unigram_nll(&st, &sv);
    let markov = naturalness::markov_nll(&st, &sv, 0.1);
    let (model, history) = naturalness::train(&st, &sv, &cfg, |e| {
        eprintln!(
            "época {:3} | pérdida {:.4} | NLL val {:.4} | exactitud {:.3}",
            e.epoch, e.train_loss, e.val_nll, e.val_acc
        );
    });
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&out, model.to_bytes())?;
    let baselines = [
        ("unigrama", uni),
        ("Markov 2D (izquierda, debajo, fila)", markov),
        ("MLP", model.nll_ref),
    ];
    save_training(&out, &history, &baselines, markov)?;
    for (n, v) in baselines {
        println!("{n:38} NLL {v:.4}  perplejidad {:.3}", v.exp());
    }
    println!(
        "modelo guardado en {} ({} parámetros)",
        out.display(),
        model.net.num_params()
    );
    Ok(())
}

/// Niveles para el agente: uno de cada cinco individuos de la población final
/// de varias ejecuciones cortas del GA. Una de cada cuatro ejecuciones se
/// reserva para la prueba, junto con niveles aleatorios nuevos. El
/// entrenamiento recorta ventanas de los niveles de entrenamiento y genera
/// niveles aleatorios sobre la marcha. Devuelve `(entrenamiento, prueba del GA, prueba aleatoria)`.
fn rl_pool(width: usize, seeds: u64) -> (Vec<Level>, Vec<Level>, Vec<Level>) {
    let checker = Checker::default();
    let mut train = Vec::new();
    let mut test = Vec::new();
    for seed in 0..seeds {
        let cfg = ga::GaConfig {
            seed: 1000 + seed,
            width,
            generations: 40,
            ..ga::GaConfig::default()
        };
        let r = ga::run(&cfg, &checker, Extras::default(), |_, _, _| {});
        let dest = if seed % 4 == 3 { &mut test } else { &mut train };
        dest.extend(r.final_population.into_iter().step_by(5).map(|(l, _)| l));
    }
    let mut rng = nivelga::rng::Rng::seed_from_u64(77);
    let random = (0..seeds as usize * 4)
        .map(|_| ga::init::random_level(width, &mut rng))
        .collect();
    (train, test, random)
}

fn entrenar_rl(mut args: Args) -> Result<()> {
    let d = rl::A2cConfig::default();
    let hidden = match args.take("ocultas") {
        Some(h) => h
            .split(',')
            .map(|v| {
                v.trim()
                    .parse()
                    .map_err(|_| Error::Config(format!("capa oculta inválida «{v}»")))
            })
            .collect::<Result<Vec<usize>>>()?,
        None => d.hidden.clone(),
    };
    let cfg = rl::A2cConfig {
        envs: args.get("entornos", d.envs)?,
        t_max: args.get("tmax", d.t_max)?,
        gamma: args.get("gamma", d.gamma)?,
        entropy: args.get("entropia", d.entropy)?,
        lr: args.get("lr", d.lr)?,
        total_steps: args.get("pasos", d.total_steps)?,
        seed: args.get("semilla", d.seed)?,
        hidden,
        ..d
    };
    let width = args.get("ancho", 60_usize)?;
    let cfg = rl::A2cConfig {
        max_width: width,
        min_width: args.get("ancho-inicial", cfg.min_width)?.min(width),
        ..cfg
    };
    let seeds = args.get("ejecuciones-ga", 32_u64)?;
    let out = PathBuf::from(
        args.take("salida")
            .unwrap_or_else(|| "modelos/agente.rla".into()),
    );
    args.finish()?;
    eprintln!("generando currículo con {seeds} ejecuciones del GA (ancho {width})...");
    let (train, test, random) = rl_pool(width, seeds);
    eprintln!(
        "{} niveles de entrenamiento y {} de prueba",
        train.len(),
        test.len()
    );
    let checker = Checker::default();
    let t0 = std::time::Instant::now();
    let (agent, history) = rl::train(&train, &checker, &cfg, |l| {
        eprintln!(
            "pasos {:8} | episodios {:5} | victorias {:5.1}% | progreso {:5.1}% | retorno {:+.3} | entropía {:.3}",
            l.steps,
            l.episodes,
            100.0 * l.win_rate,
            100.0 * l.progress,
            l.ret,
            l.entropy
        );
    });
    eprintln!("entrenamiento: {:.1} s", t0.elapsed().as_secs_f64());
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(&out, agent.to_bytes())?;
    let mut csv = String::from("pasos,episodios,victorias,progreso,retorno,entropia\n");
    for l in &history {
        let _ = writeln!(
            csv,
            "{},{},{:.4},{:.4},{:.4},{:.4}",
            l.steps, l.episodes, l.win_rate, l.progress, l.ret, l.entropy
        );
    }
    std::fs::write(out.with_extension("csv"), csv)?;
    let line = |name: &str, f: &dyn Fn(&rl::TrainLog) -> f64| nivelga::plot::Series {
        name: name.into(),
        points: history.iter().map(|l| (l.steps as f64, f(l))).collect(),
        band: vec![],
    };
    std::fs::write(
        out.with_extension("svg"),
        nivelga::plot::line_chart(
            "Agente A2C: aprendizaje",
            "pasos de entorno",
            "fracción",
            &[
                line("victorias", &|l| l.win_rate),
                line("progreso", &|l| l.progress),
                line("entropía / ln 13", &|l| l.entropy / 13f64.ln()),
            ],
        ),
    )?;
    let seen: Vec<Level> = train.iter().step_by(4).cloned().collect();
    let groups: [(&str, &[Level]); 3] = [
        ("GA, vistos (muestra)", &seen),
        ("GA, nuevos", &test),
        ("aleatorios, nuevos", &random),
    ];
    println!("{}", playtester_report(&agent, &groups, &checker, cfg.seed));
    println!(
        "agente guardado en {} ({} parámetros)",
        out.display(),
        agent.net.num_params()
    );
    Ok(())
}

/// Evalúa el agente como probador en varios grupos de niveles: victorias en
/// los jugables, comparación con la red sin entrenar, niveles injugables
/// "ganados" (debe ser 0) y correlación de Spearman con la dificultad `D`.
fn playtester_report(
    agent: &Agent,
    groups: &[(&str, &[Level])],
    checker: &Checker,
    seed: u64,
) -> String {
    let cfg = nivelga::fitness::FitnessConfig::default();
    let mut ev = nivelga::fitness::Evaluator::new(checker, &cfg);
    let untrained = Agent::new(
        agent.net.spec().hidden.clone(),
        &mut nivelga::rng::Rng::seed_from_u64(seed + 99),
    );
    let mut rep = String::from(
        "grupo | niveles | jugables (BFS) | victorias en jugables | sin entrenar | injugables ganados | Spearman(victorias, D)\n",
    );
    for (name, levels) in groups {
        let (mut base, mut false_wins) = (0.0, 0);
        let (mut rates, mut diffs) = (Vec::new(), Vec::new());
        for l in *levels {
            let e = ev.evaluate(l, Extras::default());
            let t = rl::playtest(agent, l, checker, 10, seed);
            if e.feasible {
                rates.push(t.win_rate);
                diffs.push(e.difficulty);
                base += rl::playtest(&untrained, l, checker, 10, seed).win_rate;
            } else if t.win_rate > 0.0 {
                false_wins += 1;
            }
        }
        let k = rates.len().max(1) as f64;
        let _ = writeln!(
            rep,
            "{name} | {} | {} | {:.1} % | {:.1} % | {false_wins} | {:.3}",
            levels.len(),
            rates.len(),
            100.0 * rates.iter().sum::<f64>() / k,
            100.0 * base / k,
            nivelga::stats::spearman(&rates, &diffs)
        );
    }
    rep
}

/// Guarda el historial de la PG en CSV y sus curvas (BCE y tamaño) en SVG.
fn save_gp_history(out: &std::path::Path, history: &[nivelga::gp::GpStats]) -> Result<()> {
    let mut csv = String::from("generacion,mejor_bce,tamano_medio,tamano_mejor\n");
    for h in history {
        let _ = writeln!(
            csv,
            "{},{:.6},{:.3},{}",
            h.generation, h.best_bce, h.mean_size, h.best_size
        );
    }
    std::fs::write(out.join("historial.csv"), csv)?;
    let line = |name: &str, f: &dyn Fn(&nivelga::gp::GpStats) -> f64| nivelga::plot::Series {
        name: name.into(),
        points: history
            .iter()
            .map(|h| (h.generation as f64, f(h)))
            .collect(),
        band: vec![],
    };
    std::fs::write(
        out.join("bce.svg"),
        nivelga::plot::line_chart(
            "PG: mejor cross-entropy binaria",
            "generación",
            "BCE",
            &[line("mejor BCE", &|h| h.best_bce)],
        ),
    )?;
    std::fs::write(
        out.join("tamano.svg"),
        nivelga::plot::line_chart(
            "PG: tamaño de los árboles (bloat)",
            "generación",
            "nodos",
            &[
                line("tamaño medio", &|h| h.mean_size),
                line("tamaño del mejor", &|h| h.best_size as f64),
            ],
        ),
    )?;
    Ok(())
}

/// Conjunto de datos de la PG: niveles aleatorios y variantes de niveles
/// evolucionados, etiquetados por el BFS.
fn pg_dataset(checker: &Checker, n: usize, seed: u64) -> nivelga::surrogate::Dataset {
    eprintln!("generando niveles evolucionados para el conjunto de datos...");
    let mut evolved = Vec::new();
    for s in 0..8 {
        let c = ga::GaConfig {
            seed: 500 + s,
            generations: 40,
            ..ga::GaConfig::default()
        };
        let r = ga::run(&c, checker, Extras::default(), |_, _, _| {});
        evolved.extend(r.final_population.into_iter().map(|(l, _)| l));
    }
    nivelga::surrogate::build_dataset(n, 120, &evolved, checker, seed)
}

fn entrenar_pg(mut args: Args) -> Result<()> {
    use nivelga::gp::{self, GpConfig};
    use nivelga::surrogate::{self, Logistic};
    let n = args.get("muestras", 20_000_usize)?;
    let fit_n = args.get("muestras-aptitud", 3_000_usize)?;
    let d = GpConfig::default();
    let cfg = GpConfig {
        population: args.get("poblacion", d.population)?,
        generations: args.get("generaciones", d.generations)?,
        tournament: args.get("torneo", d.tournament)?,
        parsimony: args.get("parsimonia", d.parsimony)?,
        seed: args.get("semilla", d.seed)?,
        ..d
    };
    let out = PathBuf::from(
        args.take("salida")
            .unwrap_or_else(|| "resultados/pg".into()),
    );
    args.finish()?;
    std::fs::create_dir_all(&out)?;
    let checker = Checker::default();
    let data = pg_dataset(&checker, n, cfg.seed);
    let (mut train, mut test) = data.split(4);
    let (m, sd) = train.moments();
    train.standardize(&m, &sd);
    test.standardize(&m, &sd);
    eprintln!(
        "{} muestras de entrenamiento y {} de prueba; jugables: {:.1} %",
        train.len(),
        test.len(),
        100.0 * data.positive_rate()
    );
    let threads = ga::effective_threads(0);
    // La aptitud de la PG usa una submuestra fija para que cada evaluación sea barata.
    let fit_set = surrogate::Dataset {
        x: train.x.iter().take(fit_n).copied().collect(),
        y: train.y.iter().take(fit_n).copied().collect(),
    };
    let t0 = std::time::Instant::now();
    let res = gp::evolve(&fit_set, &cfg, threads, |s| {
        if s.generation % 5 == 0 {
            eprintln!(
                "gen {:3} | BCE {:.4} | tamaño medio {:6.1} | tamaño del mejor {}",
                s.generation, s.best_bce, s.mean_size, s.best_size
            );
        }
    });
    let gp_secs = t0.elapsed().as_secs_f64();
    let logistic = Logistic::train(&train, 30, 0.01, cfg.seed);
    let mlp = surrogate::train_mlp(&train, 32, 30, 1e-3, cfg.seed);
    let mut stack = Vec::new();
    let mut tape = mlp.tape();
    let p_gp: Vec<f64> = test
        .x
        .iter()
        .map(|x| gp::predict(&res.best, res.scaling, x, &mut stack))
        .collect();
    let p_lr: Vec<f64> = test.x.iter().map(|x| logistic.predict(x)).collect();
    let p_mlp: Vec<f64> = test
        .x
        .iter()
        .map(|x| surrogate::mlp_predict(&mlp, x, &mut tape))
        .collect();
    let base = test.y.iter().sum::<f64>() / test.len().max(1) as f64;
    let p_base = vec![base; test.len()];
    let mut rep =
        String::from("modelo | BCE | exactitud | F1 injugable | AUC | ECE | parámetros\n");
    for (name, p, params) in [
        ("tasa base", &p_base, 1),
        ("regresión logística", &p_lr, surrogate::NF + 1),
        ("MLP (32 ocultas)", &p_mlp, mlp.num_params()),
        ("PG simbólica", &p_gp, res.best.len()),
    ] {
        let mt = surrogate::metrics(p, &test.y);
        let _ = writeln!(
            rep,
            "{name} | {:.4} | {:.3} | {:.3} | {:.3} | {:.3} | {params}",
            mt.bce, mt.accuracy, mt.f1_unplayable, mt.auc, mt.ece
        );
    }
    let _ = writeln!(
        rep,
        "\nPG: {gp_secs:.1} s, {} nodos, profundidad {}",
        res.best.len(),
        gp::depth(&res.best)
    );
    let _ = writeln!(
        rep,
        "jugable ⇔ σ({:.3}·f + {:.3}) ≥ 0,5 con rasgos estandarizados y\nf = {}",
        res.scaling.0,
        res.scaling.1,
        gp::to_infix(&res.best)
    );
    std::fs::write(out.join("informe.txt"), &rep)?;
    save_gp_history(&out, &res.history)?;
    println!("{rep}");
    Ok(())
}

/// Semilla a partir del reloj (para generar niveles distintos cada vez).
fn clock_seed() -> u64 {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    nivelga::rng::mix_seed(&[
        t.as_secs(),
        u64::from(t.subsec_nanos()),
        u64::from(std::process::id()),
    ])
}

/// Documento JSON de un nivel generado para el juego en Godot.
fn level_json(
    level: &Level,
    e: &nivelga::fitness::Evaluation,
    traj: &[(usize, usize)],
    assets: &std::path::Path,
    background: &str,
    (seed, target, seconds): (u64, f64, f64),
) -> nivelga::json::Json {
    use nivelga::json::Json;
    let abs = |p: &std::path::Path| {
        std::fs::canonicalize(p)
            .map_or_else(|_| p.display().to_string(), |a| a.display().to_string())
    };
    let rows: Vec<String> = (0..level.height())
        .map(|y| {
            (0..level.width())
                .map(|x| level.get(x, y).to_char())
                .collect()
        })
        .collect();
    let f = &e.features;
    let metrics = Json::obj()
        .with("aptitud", e.fitness)
        .with("jugable", e.feasible)
        .with("calidad", e.quality)
        .with("dificultad", e.difficulty)
        .with("variedad", e.variety)
        .with("coherencia", e.coherence)
        .with("naturalidad", e.naturalness)
        .with("saltos", f.jumps)
        .with("saltos_ajustados", f.tight_jumps)
        .with("enemigos", f.enemies)
        .with("peligros", f.hazards)
        .with("huecos", f.gaps)
        .with("monedas", f.coins)
        .with("escaleras", f.ladders)
        .with("linealidad", f.linearity)
        .with("leniencia", f.leniency);
    let mut sprites = Json::obj();
    for (name, refs) in nivelga::render::dynamic_sprites(level.style) {
        let list: Vec<Json> = refs
            .iter()
            .map(|r| {
                Json::obj()
                    .with("hoja", abs(&assets.join(r.file)))
                    .with("indice", u32::from(r.index))
                    .with("columnas", r.columns)
                    .with("tam", r.size)
            })
            .collect();
        sprites = sprites.with(name, Json::Arr(list));
    }
    Json::obj()
        .with("version", 1_u32)
        .with("semilla", seed.to_string())
        .with("ancho", level.width())
        .with("alto", level.height())
        .with("estilo", level.style.name())
        .with("dificultad_objetivo", target)
        .with("segundos_generacion", seconds)
        .with("fondo", background)
        .with("filas", rows)
        .with(
            "trayectoria",
            Json::Arr(traj.iter().map(|&(x, y)| Json::from(vec![x, y])).collect()),
        )
        .with("metricas", metrics)
        .with("sprites", sprites)
}

/// Genera un nivel nuevo para el juego en Godot: evoluciona con el GA (crowding
/// con torneo, el mejor en E11) y escribe `nivel.json` y `fondo.png`.
fn generar(mut args: Args) -> Result<()> {
    let seed = match args.take("semilla") {
        Some(v) => v
            .parse()
            .map_err(|_| Error::Config(format!("semilla inválida «{v}»")))?,
        None => clock_seed(),
    };
    let mut rng = nivelga::rng::Rng::seed_from_u64(seed);
    let width = args.get("ancho", 60 + rng.index(51))?;
    let target = args.get("dificultad", 0.3 + 0.4 * rng.f64())?;
    // Por defecto el estilo se sortea y se fija (variedad visual en el juego);
    // con `--estilo libre` lo elige el GA, que tiende a converger a pradera.
    let fixed_style = match args.take("estilo").as_deref() {
        None | Some("azar") => {
            Some(nivelga::level::Style::ALL[rng.index(nivelga::level::Style::COUNT)])
        }
        Some("libre") => None,
        Some(n) => Some(
            nivelga::level::Style::from_name(n)
                .ok_or_else(|| Error::Config(format!("estilo desconocido «{n}»")))?,
        ),
    };
    let generations = args.get("generaciones", 80_usize)?;
    let population = args.get("poblacion", 60_usize)?;
    let out = PathBuf::from(
        args.take("salida")
            .unwrap_or_else(|| "resultados/juego/actual".into()),
    );
    let assets = PathBuf::from(args.take("recursos").unwrap_or_else(|| "assets".into()));
    let ext = Externals::load(&mut args)?;
    args.finish()?;
    let checker = Checker::default();
    let t0 = std::time::Instant::now();
    let mut cfg = ga::GaConfig {
        seed,
        width,
        generations,
        population,
        niching: ga::Niching::CrowdingSelective,
        fixed_style,
        threads: 1,
        ..ga::GaConfig::default()
    };
    cfg.fitness.targets.difficulty = target;
    if ext.model.is_some() {
        cfg.fitness.weights.naturalness = 0.15;
    }
    let mut result = ga::run(&cfg, &checker, ext.extras(), |_, _, _| {});
    // Si por mala suerte el mejor no es jugable, se reintenta con otra semilla.
    for k in 1..4 {
        if result.best_eval.feasible {
            break;
        }
        cfg.seed = seed.wrapping_add(k);
        result = ga::run(&cfg, &checker, ext.extras(), |_, _, _| {});
    }
    let level = &result.best;
    let play = checker.check(level, &mut PlayScratch::default());
    let traj = checker.trajectory(level, &play);
    std::fs::create_dir_all(&out)?;
    let sets = Tilesets::load(&assets)?;
    let bg_path = out.join("fondo.png");
    nivelga::image::png::save(&nivelga::render::render_static(level, &sets), &bg_path)?;
    std::fs::write(out.join("nivel.txt"), level.to_string())?;
    let abs = |p: &std::path::Path| {
        std::fs::canonicalize(p)
            .map_or_else(|_| p.display().to_string(), |a| a.display().to_string())
    };
    let doc = level_json(
        &result.best,
        &result.best_eval,
        &traj,
        &assets,
        &abs(&bg_path),
        (seed, target, t0.elapsed().as_secs_f64()),
    );
    let json_path = out.join("nivel.json");
    std::fs::write(&json_path, doc.dump())?;
    println!("{}", abs(&json_path));
    Ok(())
}

/// Convierte un nivel en texto al formato del juego (JSON + PNG).
fn exportar(mut args: Args) -> Result<()> {
    let level = load_level(&args)?;
    let out = PathBuf::from(
        args.take("salida")
            .unwrap_or_else(|| "resultados/juego/exportado".into()),
    );
    let assets = PathBuf::from(args.take("recursos").unwrap_or_else(|| "assets".into()));
    args.finish()?;
    let checker = Checker::default();
    let cfg = nivelga::fitness::FitnessConfig::default();
    let e = nivelga::fitness::Evaluator::new(&checker, &cfg).evaluate(&level, Extras::default());
    let play = checker.check(&level, &mut PlayScratch::default());
    let traj = checker.trajectory(&level, &play);
    std::fs::create_dir_all(&out)?;
    let bg = out.join("fondo.png");
    nivelga::image::png::save(
        &nivelga::render::render_static(&level, &Tilesets::load(&assets)?),
        &bg,
    )?;
    let bg_abs = std::fs::canonicalize(&bg)?.display().to_string();
    let doc = level_json(
        &level,
        &e,
        &traj,
        &assets,
        &bg_abs,
        (0, cfg.targets.difficulty, 0.0),
    );
    let path = out.join("nivel.json");
    std::fs::write(&path, doc.dump())?;
    println!("{}", std::fs::canonicalize(&path)?.display());
    Ok(())
}

/// Analiza el registro de partidas del juego en Godot: compara la dificultad
/// medida por el GA (`D`) con la experiencia de quien juega.
fn partidas(args: &Args) -> Result<()> {
    let path = args
        .positional
        .first()
        .cloned()
        .unwrap_or_else(|| "resultados/partidas.csv".into());
    args.finish()?;
    let text = std::fs::read_to_string(&path)?;
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap_or_default().split(',').collect();
    let col = |name: &str| {
        header.iter().position(|h| *h == name).ok_or_else(|| {
            Error::format("registro de partidas", format!("falta la columna {name}"))
        })
    };
    let (cd, cr, cm, cs, cmk) = (
        col("dificultad")?,
        col("resultado")?,
        col("muertes")?,
        col("segundos")?,
        col("marcado")?,
    );
    let (mut d, mut deaths, mut secs, mut won_d, mut lost_d) =
        (vec![], vec![], vec![], vec![], vec![]);
    let (mut won, mut skipped, mut flagged) = (0, 0, 0);
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let f: Vec<&str> = line.split(',').collect();
        let num = |i: usize| {
            f.get(i)
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(f64::NAN)
        };
        let dif = num(cd);
        d.push(dif);
        deaths.push(num(cm));
        secs.push(num(cs));
        if f.get(cr) == Some(&"ganado") {
            won += 1;
            won_d.push(dif);
        } else {
            skipped += 1;
            lost_d.push(dif);
        }
        flagged += usize::from(f.get(cmk) == Some(&"si"));
    }
    let n = d.len();
    println!(
        "{n} partidas: {won} superadas, {skipped} saltadas o abandonadas, {flagged} marcadas para revisar"
    );
    if n >= 3 {
        println!(
            "Spearman(D, muertes)  = {:+.3}",
            nivelga::stats::spearman(&d, &deaths)
        );
        println!(
            "Spearman(D, segundos) = {:+.3}",
            nivelga::stats::spearman(&d, &secs)
        );
    }
    if !won_d.is_empty() && !lost_d.is_empty() {
        let mw = nivelga::stats::mann_whitney(&lost_d, &won_d);
        println!(
            "D mediana de los superados {:.3} frente a los no superados {:.3} (Mann-Whitney p = {:.3e}, Â12 = {:.3})",
            nivelga::stats::summary(&won_d).median,
            nivelga::stats::summary(&lost_d).median,
            mw.p,
            nivelga::stats::a12(&lost_d, &won_d)
        );
    }
    Ok(())
}

/// Mide los rasgos estructurales de un corpus de niveles diseñados por
/// personas (VGLC) para calibrar los objetivos de la aptitud.
fn corpus(mut args: Args) -> Result<()> {
    use nivelga::metrics::{Features, MetricScratch};
    let data = args.take("datos").unwrap_or_else(|| "data/vglc/smb".into());
    args.finish()?;
    let levels = nivelga::vglc::load_dir(&data)?;
    let mut ms = MetricScratch::default();
    let sc = PlayScratch::default();
    let play = nivelga::play::PlayResult::default();
    let mut rows: Vec<[f64; 9]> = Vec::new();
    println!(
        "nivel | ancho | patrones 3×3 | entropía | plataformas/col | monedas/col | enemigos/col | huecos/col | densidad | linealidad"
    );
    for (name, l) in &levels {
        let f = Features::compute(l, &play, &sc, &mut ms);
        let w = l.width() as f64;
        let r = [
            w,
            f.pattern_ratio,
            f.entropy,
            f64::from(f.platforms) / w,
            f64::from(f.coins) / w,
            f64::from(f.enemies) / w,
            f64::from(f.gaps) / w,
            f.density,
            f.linearity,
        ];
        println!(
            "{name} | {:.0} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3}",
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8]
        );
        rows.push(r);
    }
    let names = [
        "ancho",
        "patrones",
        "entropía",
        "plataformas/col",
        "monedas/col",
        "enemigos/col",
        "huecos/col",
        "densidad",
        "linealidad",
    ];
    println!();
    for (k, n) in names.iter().enumerate() {
        let v: Vec<f64> = rows.iter().map(|r| r[k]).collect();
        let s = nivelga::stats::summary(&v);
        println!(
            "{n:16} mediana {:.3}  [Q1 {:.3}; Q3 {:.3}]  rango [{:.3}; {:.3}]",
            s.median, s.q1, s.q3, s.min, s.max
        );
    }
    Ok(())
}

fn jugar(mut args: Args) -> Result<()> {
    let level = load_level(&args)?;
    let path = args
        .take("agente")
        .unwrap_or_else(|| "modelos/agente.rla".into());
    let cols = args.get("columnas", 60_usize)?;
    let delay = args.get("retardo", 80_u64)?;
    let seed = args.get("semilla", 0_u64)?;
    let assets = args.take("recursos").unwrap_or_else(|| "assets".into());
    args.finish()?;
    let agent = Agent::from_bytes(&std::fs::read(path)?)?;
    let checker = Checker::default();
    let (traj, won) = agent.trajectory(
        &level,
        &checker,
        &mut nivelga::rng::Rng::seed_from_u64(seed),
    );
    ansi::animate(
        &level,
        &Tilesets::load(assets)?,
        &traj,
        cols.min(level.width()),
        delay,
    )?;
    println!(
        "{}",
        if won {
            "el agente llegó a la meta"
        } else {
            "el agente no llegó a la meta"
        }
    );
    Ok(())
}

fn experimento(mut args: Args) -> Result<()> {
    use nivelga::experiment;
    let which = args
        .positional
        .first()
        .cloned()
        .ok_or_else(|| Error::Config("indica el experimento (e1..e10 o todos)".into()))?;
    let runs = args.get("repeticiones", 30_u64)?;
    let threads = ga::effective_threads(args.get("hilos-experimento", 0_usize)?);
    let out = PathBuf::from(
        args.take("salida")
            .unwrap_or_else(|| "resultados/experimentos".into()),
    );
    let base = cli::ga_config(&mut args)?;
    let ext = Externals::load(&mut args)?;
    args.finish()?;
    let scorer = ext.model.as_ref().map(|m| m as &dyn Scorer);
    let names: Vec<&str> = if which == "todos" {
        experiment::EXPERIMENTS.iter().map(|e| e.0).collect()
    } else {
        vec![which.as_str()]
    };
    for name in names {
        if name == "e8" && scorer.is_none() {
            eprintln!("e8 necesita --naturalidad RUTA (ver `nivelga entrenar-nn`); se omite");
            continue;
        }
        let variants = experiment::define(name, &base)?;
        let title = experiment::EXPERIMENTS
            .iter()
            .find(|e| e.0 == name)
            .map_or(name, |e| e.1);
        eprintln!(
            "{name}: {title} ({} variantes × {runs} semillas, {threads} hilos)",
            variants.len()
        );
        let t0 = std::time::Instant::now();
        let results = experiment::run_all(&variants, runs, threads, ext.extras(), |k, total| {
            if k % 10 == 0 || k == total {
                eprint!("\r  {k}/{total} ejecuciones");
            }
        });
        eprintln!(" en {:.1} s", t0.elapsed().as_secs_f64());
        let dir = out.join(name);
        let mut rep = experiment::analyze(&dir, title, &variants, &results)?;
        if name == "e2" {
            rep.push_str(&experiment::e2_selection_pressure(&variants, &results));
            std::fs::write(dir.join("informe.txt"), &rep)?;
        }
        if name == "e5" {
            experiment::e5_heatmap(&dir, &variants, &results)?;
        }
        if let Some(m) = scorer
            && name == "e8"
        {
            rep.push_str(&experiment::e8_naturalness(&variants, &results, m));
            std::fs::write(dir.join("informe.txt"), &rep)?;
        }
        if name == "e9" {
            rep.push_str(&experiment::e9_expressive_range(&dir, &results)?);
            std::fs::write(dir.join("informe.txt"), &rep)?;
        }
        println!("{rep}");
    }
    Ok(())
}

fn load_level(args: &Args) -> Result<Level> {
    let path = args
        .positional
        .first()
        .ok_or_else(|| Error::Config("falta la ruta del nivel".into()))?;
    Level::parse(&std::fs::read_to_string(path)?)
}

fn ver(mut args: Args) -> Result<()> {
    let level = load_level(&args)?;
    let cols = args.get("columnas", 60_usize)?;
    let delay = args.get("retardo", 60_u64)?;
    let assets = args.take("recursos").unwrap_or_else(|| "assets".into());
    args.finish()?;
    let checker = Checker::default();
    let res = checker.check(&level, &mut PlayScratch::default());
    if !res.solvable {
        println!(
            "el nivel no es jugable (progreso {:.0} %)",
            100.0 * res.progress
        );
    }
    let traj = checker.trajectory(&level, &res);
    ansi::animate(
        &level,
        &Tilesets::load(assets)?,
        &traj,
        cols.min(level.width()),
        delay,
    )?;
    Ok(())
}

fn dibujar(mut args: Args) -> Result<()> {
    let level = load_level(&args)?;
    let out = args
        .positional
        .get(1)
        .cloned()
        .ok_or_else(|| Error::Config("falta la ruta de salida".into()))?;
    let scale = args.get("escala", 2_u32)?;
    let assets = args.take("recursos").unwrap_or_else(|| "assets".into());
    args.finish()?;
    let img =
        report::render_with_path(&level, &Checker::default(), &Tilesets::load(assets)?, scale);
    nivelga::image::png::save(&img, out)
}

fn comprobar(mut args: Args) -> Result<()> {
    let level = load_level(&args)?;
    let ext = Externals::load(&mut args)?;
    args.finish()?;
    let checker = Checker::default();
    let mut cfg = nivelga::fitness::FitnessConfig::default();
    cfg.weights.naturalness = if ext.model.is_some() { 0.15 } else { 0.0 };
    cfg.weights.playtester = if ext.agent.is_some() { 0.15 } else { 0.0 };
    let mut ev = nivelga::fitness::Evaluator::new(&checker, &cfg);
    println!("{}", report::describe(&ev.evaluate(&level, ext.extras())));
    if let Some(a) = &ext.agent {
        let t = rl::playtest(&a.agent, &level, &checker, 20, 0);
        println!(
            "agente: victorias {:.0} %, progreso medio {:.0} %, pasos medios {:.0}",
            100.0 * t.win_rate,
            100.0 * t.progress,
            t.steps
        );
    }
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;
    match args.command.as_str() {
        "evolucionar" => evolucionar(args),
        "ver" => ver(args),
        "dibujar" => dibujar(args),
        "comprobar" => comprobar(args),
        "experimento" => experimento(args),
        "entrenar-nn" => entrenar_nn(args),
        "entrenar-rl" => entrenar_rl(args),
        "jugar" => jugar(args),
        "entrenar-pg" => entrenar_pg(args),
        "generar" => generar(args),
        "exportar" => exportar(args),
        "partidas" => partidas(&args),
        "corpus" => corpus(args),
        "experimentos" => {
            for (n, d) in nivelga::experiment::EXPERIMENTS {
                println!("{n:4}  {d}");
            }
            Ok(())
        }
        "" | "ayuda" | "--help" | "-h" => {
            print!("{AYUDA}");
            Ok(())
        }
        other => Err(Error::Config(format!(
            "orden desconocida «{other}»; prueba `nivelga ayuda`"
        ))),
    }
}
