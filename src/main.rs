//! Interfaz de línea de órdenes de `nivelga`.

use nivelga::cli::{self, Args};
use nivelga::ga;
use nivelga::level::Level;
use nivelga::naturalness::NaturalnessModel;
use nivelga::play::{Checker, PlayScratch};
use nivelga::render::Tilesets;
use nivelga::{Error, Result, ansi, report};
use std::path::PathBuf;

const AYUDA: &str = "\
nivelga: niveles de plataformas 2D evolucionados con un algoritmo genético

USO:
  nivelga evolucionar [opciones]      ejecuta el GA y guarda resultados
  nivelga ver <nivel.txt> [--columnas 60] [--retardo 60]
                                      anima en la terminal el camino óptimo
  nivelga dibujar <nivel.txt> <salida.png> [--escala 2]
  nivelga comprobar <nivel.txt>       analiza jugabilidad y métricas
  nivelga experimento <e1..e10|todos> [--repeticiones 30] [--salida DIR]
                                      [opciones de la configuración base]
  nivelga experimentos                lista los experimentos disponibles
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
  --agregacion A       suma | geometrica | penalizacion:λ | deb  (suma)
  --dificultad D       dificultad objetivo D* (0.5)
  --hilos N            hilos de evaluación (0 = todos)
  --naturalidad RUTA   añade el término de la red de naturalidad
  --peso-nat W         peso de ese término (0.15)
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
    let model = load_model(&mut args)?;
    let mut cfg = cfg;
    if model.is_some() {
        cfg.fitness.weights.naturalness = args.get("peso-nat", 0.15)?;
    }
    args.finish()?;
    let checker = Checker::default();
    let scorer = model.as_ref().map(|m| m as &dyn nivelga::fitness::Scorer);
    let result = ga::run(&cfg, &checker, scorer, |s, _, _| {
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

/// Carga el modelo de naturalidad si se pasó `--naturalidad RUTA`.
fn load_model(args: &mut Args) -> Result<Option<NaturalnessModel>> {
    args.take("naturalidad")
        .map(|p| NaturalnessModel::from_bytes(&std::fs::read(p)?))
        .transpose()
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
    let model = load_model(&mut args)?;
    args.finish()?;
    let scorer = model.as_ref().map(|m| m as &dyn nivelga::fitness::Scorer);
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
        let results = experiment::run_all(&variants, runs, threads, scorer, |k, total| {
            if k % 10 == 0 || k == total {
                eprint!("\r  {k}/{total} ejecuciones");
            }
        });
        eprintln!(" en {:.1} s", t0.elapsed().as_secs_f64());
        let dir = out.join(name);
        let mut rep = experiment::analyze(&dir, title, &variants, &results)?;
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

fn comprobar(args: &Args) -> Result<()> {
    let level = load_level(args)?;
    args.finish()?;
    let checker = Checker::default();
    let cfg = nivelga::fitness::FitnessConfig::default();
    let mut ev = nivelga::fitness::Evaluator::new(&checker, &cfg);
    println!("{}", report::describe(&ev.evaluate(&level, None)));
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse(std::env::args().skip(1))?;
    match args.command.as_str() {
        "evolucionar" => evolucionar(args),
        "ver" => ver(args),
        "dibujar" => dibujar(args),
        "comprobar" => comprobar(&args),
        "experimento" => experimento(args),
        "entrenar-nn" => entrenar_nn(args),
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
