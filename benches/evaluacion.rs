//! Medición del coste de cada pieza de la evaluación sin dependencias:
//! `std::time::Instant` con `std::hint::black_box` y la mediana de varias
//! repeticiones (el atributo `#[bench]` sigue siendo inestable).
//!
//! Ejecutar con `cargo bench --bench evaluacion`.

use nivelga::fitness::{Evaluator, FitnessConfig, Scorer};
use nivelga::ga::init::random_level;
use nivelga::ga::selection::{Scaling, roulette, scale, sus};
use nivelga::level::Level;
use nivelga::metrics::{Features, MetricScratch};
use nivelga::naturalness::NaturalnessModel;
use nivelga::play::{Checker, PlayScratch};
use nivelga::rng::Rng;
use std::hint::black_box;
use std::time::Instant;

/// Mide `f` sobre `items` y devuelve la mediana de microsegundos por elemento.
fn bench<T>(name: &str, items: &[T], reps: usize, mut f: impl FnMut(&T)) -> f64 {
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for it in items {
            f(black_box(it));
        }
        times.push(t0.elapsed().as_secs_f64() * 1e6 / items.len() as f64);
    }
    times.sort_by(f64::total_cmp);
    let med = times[reps / 2];
    println!("{name:40} {med:10.2} µs");
    med
}

fn main() {
    let mut rng = Rng::seed_from_u64(42);
    let levels: Vec<Level> = (0..200).map(|_| random_level(120, &mut rng)).collect();
    let checker = Checker::default();
    let cfg = FitnessConfig::default();
    let mut scratch = PlayScratch::default();
    let mut ms = MetricScratch::default();
    let mut ev = Evaluator::new(&checker, &cfg);

    bench("verificador (Dijkstra + arcos)", &levels, 9, |l| {
        black_box(checker.check(l, &mut scratch));
    });
    let plays: Vec<_> = levels
        .iter()
        .map(|l| (l, checker.check(l, &mut scratch)))
        .collect();
    bench("rasgos y métricas", &plays, 9, |(l, p)| {
        black_box(Features::compute(l, p, &scratch, &mut ms));
    });
    bench("evaluación completa", &levels, 9, |l| {
        black_box(ev.evaluate(l, None));
    });
    if let Ok(bytes) = std::fs::read("modelos/naturalidad.nng")
        && let Ok(model) = NaturalnessModel::from_bytes(&bytes)
    {
        bench("naturalidad (MLP por tile)", &levels, 5, |l| {
            black_box(model.score(l));
        });
    } else {
        println!("(sin modelos/naturalidad.nng: se omite la red)");
    }

    let fitness: Vec<f64> = (0..1000).map(|_| rng.f64()).collect();
    let pops = vec![fitness; 50];
    let (mut scaled, mut prefix, mut out) = (Vec::new(), Vec::new(), Vec::new());
    let mut r2 = Rng::seed_from_u64(1);
    bench("ruleta: 1000 giros (búsqueda binaria)", &pops, 9, |f| {
        scale(f, Scaling::Linear(1.5), &mut scaled);
        out.clear();
        roulette(&scaled, 1000, &mut r2, &mut prefix, &mut out);
        black_box(&out);
    });
    bench("SUS: 1000 punteros", &pops, 9, |f| {
        scale(f, Scaling::Linear(1.5), &mut scaled);
        out.clear();
        sus(&scaled, 1000, &mut r2, &mut out);
        black_box(&out);
    });
}
