//! Programación Genética (Koza, 1992; Poli, Langdon y McPhee, *A Field Guide
//! to Genetic Programming*, 2008) para aprender un **clasificador simbólico**
//! de jugabilidad a partir de rasgos estructurales.
//!
//! - Representación lineal en prefijo (`Vec<Node>`), como TinyGP: un subárbol
//!   es un tramo contiguo, así el cruce es un empalme de slices.
//! - Primitivas: `+ − × ÷` (división protegida que devuelve 1), `min`, `max` y
//!   `si_mayor(a, b, c, d)` (= c si a > b, si no d); terminales: rasgos y
//!   constantes efímeras en `[−5, 5]`.
//! - Inicialización *ramped half-and-half* (profundidades 2 a 6).
//! - Aptitud a minimizar: cross-entropy binaria de `σ(a·árbol(x) + b)` frente
//!   a la etiqueta del BFS, con `(a, b)` ajustados por escalado lineal
//!   (Keijzer, 2003), más una presión de parsimonia `c · tamaño` (control del
//!   *bloat*) y límites duros de tamaño y profundidad.
//! - Selección por torneo, cruce de subárbol con sesgo 90/10 hacia nodos
//!   internos y mutación de subárbol o puntual.

use crate::nn::loss::sigmoid;
use crate::rng::Rng;
use crate::surrogate::{Dataset, FEATURE_NAMES, NF};
use std::fmt::Write as _;

/// Nodo de un árbol en prefijo.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Node {
    /// Suma.
    Add,
    /// Resta.
    Sub,
    /// Producto.
    Mul,
    /// División protegida (devuelve 1 si el divisor es casi 0).
    Div,
    /// Mínimo.
    Min,
    /// Máximo.
    Max,
    /// `si_mayor(a, b, c, d)`.
    IfGt,
    /// Rasgo de entrada.
    Feat(u8),
    /// Constante.
    Const(f32),
}

impl Node {
    /// Aridad.
    pub const fn arity(self) -> usize {
        match self {
            Self::Add | Self::Sub | Self::Mul | Self::Div | Self::Min | Self::Max => 2,
            Self::IfGt => 4,
            Self::Feat(_) | Self::Const(_) => 0,
        }
    }
}

const FUNCTIONS: [Node; 7] = [
    Node::Add,
    Node::Sub,
    Node::Mul,
    Node::Div,
    Node::Min,
    Node::Max,
    Node::IfGt,
];

/// Árbol de expresión en prefijo.
pub type Tree = Vec<Node>;

/// Índice final (exclusivo) del subárbol que empieza en `start`: se recorre
/// sumando `aridad − 1` a un contador que empieza en 1 hasta llegar a 0.
pub fn subtree_end(t: &[Node], start: usize) -> usize {
    let mut need = 1_i64;
    let mut i = start;
    while need > 0 {
        need += t[i].arity() as i64 - 1;
        i += 1;
    }
    i
}

/// Profundidad del árbol.
pub fn depth(t: &[Node]) -> usize {
    // Pila de "hijos restantes" por nivel abierto.
    let mut stack: Vec<usize> = Vec::new();
    let mut max = 0;
    for n in t {
        stack.push(n.arity());
        max = max.max(stack.len());
        while let Some(&0) = stack.last() {
            stack.pop();
            if let Some(top) = stack.last_mut() {
                *top -= 1;
            } else {
                break;
            }
        }
    }
    max
}

/// Evalúa el árbol con una pila recorriendo el prefijo de derecha a izquierda.
pub fn eval(t: &[Node], x: &[f64; NF], stack: &mut Vec<f64>) -> f64 {
    stack.clear();
    for &n in t.iter().rev() {
        let v = match n {
            Node::Feat(i) => x[usize::from(i)],
            Node::Const(c) => f64::from(c),
            Node::IfGt => {
                let (a, b, c, d) = (pop(stack), pop(stack), pop(stack), pop(stack));
                if a > b { c } else { d }
            }
            f => {
                let (a, b) = (pop(stack), pop(stack));
                match f {
                    Node::Add => a + b,
                    Node::Sub => a - b,
                    Node::Mul => a * b,
                    Node::Div => {
                        if b.abs() < 1e-9 {
                            1.0
                        } else {
                            a / b
                        }
                    }
                    Node::Min => a.min(b),
                    _ => a.max(b),
                }
            }
        };
        stack.push(if v.is_finite() {
            v.clamp(-1e6, 1e6)
        } else {
            0.0
        });
    }
    stack.pop().unwrap_or(0.0)
}

fn pop(s: &mut Vec<f64>) -> f64 {
    s.pop().unwrap_or(0.0)
}

fn terminal(rng: &mut Rng) -> Node {
    if rng.chance(0.75) {
        Node::Feat(rng.index(NF) as u8)
    } else {
        Node::Const((rng.f64() * 10.0 - 5.0) as f32)
    }
}

/// Árbol aleatorio por el método *full* (`full = true`) o *grow*.
pub fn random_tree(depth_left: usize, full: bool, rng: &mut Rng, out: &mut Tree) {
    let pick_terminal = depth_left <= 1 || (!full && rng.chance(0.3));
    if pick_terminal {
        out.push(terminal(rng));
    } else {
        let f = FUNCTIONS[rng.index(FUNCTIONS.len())];
        out.push(f);
        for _ in 0..f.arity() {
            random_tree(depth_left - 1, full, rng, out);
        }
    }
}

/// Parámetros de la PG (valores de la *Field Guide* salvo donde se indica).
#[derive(Debug, Clone)]
pub struct GpConfig {
    /// Tamaño de población.
    pub population: usize,
    /// Generaciones.
    pub generations: usize,
    /// Tamaño de torneo.
    pub tournament: usize,
    /// Probabilidad de cruce de subárbol.
    pub p_crossover: f64,
    /// Probabilidad de mutación (de subárbol o puntual).
    pub p_mutation: f64,
    /// Tamaño máximo (nodos).
    pub max_size: usize,
    /// Profundidad máxima.
    pub max_depth: usize,
    /// Coeficiente de parsimonia por nodo.
    pub parsimony: f64,
    /// Élites.
    pub elitism: usize,
    /// Semilla.
    pub seed: u64,
}

impl Default for GpConfig {
    fn default() -> Self {
        Self {
            population: 500,
            generations: 50,
            tournament: 7,
            p_crossover: 0.9,
            p_mutation: 0.1,
            max_size: 128,
            max_depth: 17,
            parsimony: 1e-3,
            elitism: 2,
            seed: 0,
        }
    }
}

/// Estadísticas de una generación de la PG.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpStats {
    /// Generación.
    pub generation: usize,
    /// Menor BCE de la población (sin parsimonia).
    pub best_bce: f64,
    /// Tamaño medio de los árboles (para observar el *bloat*).
    pub mean_size: f64,
    /// Tamaño del mejor árbol.
    pub best_size: usize,
}

/// Ajusta `(a, b)` para minimizar la BCE de `σ(a·z + b)` sobre las salidas `z`
/// de un árbol: es el *escalado lineal* de Keijzer (2003) adaptado a la
/// clasificación, resuelto con pasos de Newton recortados (regresión
/// logística 1D con una pequeña regularización L2).
/// Así la evolución busca la forma de la expresión y no su escala.
/// Devuelve `(a, b, BCE)`.
pub fn fit_scaling(z: &[f64], y: &[f64]) -> (f64, f64, f64) {
    // Regularización L2 pequeña: mantiene la solución finita con datos separables.
    const L2: f64 = 1e-3;
    let n = z.len().max(1) as f64;
    // Las salidas se reescalan a [−1, 1] para que Newton sea estable.
    let zs = z.iter().map(|v| v.abs()).fold(0.0_f64, f64::max).max(1e-9);
    let (mut a, mut b) = (0.0_f64, 0.0_f64);
    for _ in 0..30 {
        let (mut ga, mut gb, mut haa, mut hab, mut hbb) =
            (L2 * a * n, L2 * b * n, L2 * n, 0.0, L2 * n);
        for (&zi, &yi) in z.iter().zip(y) {
            let u = zi / zs;
            let p = sigmoid(a * u + b);
            let r = p - yi;
            let w = p * (1.0 - p);
            ga += r * u;
            gb += r;
            haa += w * u * u;
            hab += w * u;
            hbb += w;
        }
        let det = haa * hbb - hab * hab;
        if det.abs() < 1e-12 {
            break;
        }
        // Paso de Newton recortado para que no se dispare en las primeras iteraciones.
        let da = ((hbb * ga - hab * gb) / det).clamp(-4.0, 4.0);
        let db = ((haa * gb - hab * ga) / det).clamp(-4.0, 4.0);
        a = (a - da).clamp(-1e3, 1e3);
        b = (b - db).clamp(-1e3, 1e3);
        if da.abs() + db.abs() < 1e-8 {
            break;
        }
    }
    let loss = z
        .iter()
        .zip(y)
        .map(|(&zi, &yi)| crate::nn::loss::bce_with_logits(a * zi / zs + b, yi, 1.0).0)
        .sum::<f64>()
        / n;
    (a / zs, b, loss)
}

/// BCE media de un árbol sobre un conjunto de datos (sin escalado).
pub fn bce(t: &[Node], d: &Dataset, stack: &mut Vec<f64>) -> f64 {
    let mut s = 0.0;
    for (x, &y) in d.x.iter().zip(&d.y) {
        let z = eval(t, x, stack);
        let (l, _) = crate::nn::loss::bce_with_logits(z, y, 1.0);
        s += l;
    }
    s / d.len().max(1) as f64
}

/// Probabilidad de jugable según el árbol y su escalado `(a, b)`.
pub fn predict(t: &[Node], scaling: (f64, f64), x: &[f64; NF], stack: &mut Vec<f64>) -> f64 {
    sigmoid(scaling.0 * eval(t, x, stack) + scaling.1)
}

/// Elige un punto de corte con sesgo 90/10 hacia nodos internos.
fn pick_point(t: &[Node], rng: &mut Rng) -> usize {
    let internal = t.iter().any(|n| n.arity() > 0);
    let want_internal = internal && rng.chance(0.9);
    loop {
        let i = rng.index(t.len());
        if (t[i].arity() > 0) == want_internal {
            return i;
        }
    }
}

/// Cruce de subárbol: sustituye un subárbol de `a` por uno de `b`.
fn crossover(a: &[Node], b: &[Node], rng: &mut Rng, out: &mut Tree) {
    let i = pick_point(a, rng);
    let j = pick_point(b, rng);
    let (ie, je) = (subtree_end(a, i), subtree_end(b, j));
    out.clear();
    out.extend_from_slice(&a[..i]);
    out.extend_from_slice(&b[j..je]);
    out.extend_from_slice(&a[ie..]);
}

/// Mutación: de subárbol (50 %) o puntual (cambia un nodo por otro de igual aridad).
fn mutate(a: &[Node], rng: &mut Rng, out: &mut Tree) {
    out.clear();
    out.extend_from_slice(a);
    let i = rng.index(out.len());
    if rng.chance(0.5) {
        let end = subtree_end(out, i);
        let mut sub = Vec::new();
        random_tree(1 + rng.index(4), false, rng, &mut sub);
        out.splice(i..end, sub);
    } else {
        out[i] = match out[i] {
            Node::Feat(_) | Node::Const(_) => terminal(rng),
            Node::IfGt => Node::IfGt,
            _ => FUNCTIONS[rng.index(FUNCTIONS.len() - 1)],
        };
    }
}

/// Resultado de una ejecución de la PG.
#[derive(Debug, Clone)]
pub struct GpResult {
    /// Mejor árbol (por aptitud con parsimonia).
    pub best: Tree,
    /// Escalado `(a, b)` del mejor árbol: `p = σ(a·f(x) + b)`.
    pub scaling: (f64, f64),
    /// Historial.
    pub history: Vec<GpStats>,
}

/// Evoluciona un clasificador sobre `train` (evaluación paralela por hilos).
pub fn evolve(
    train: &Dataset,
    cfg: &GpConfig,
    threads: usize,
    mut log: impl FnMut(&GpStats),
) -> GpResult {
    let mut rng = Rng::seed_from_u64(cfg.seed);
    // Ramped half-and-half: profundidades 2..=6, mitad full y mitad grow.
    let mut pop: Vec<Tree> = (0..cfg.population)
        .map(|i| {
            let mut t = Vec::new();
            random_tree(2 + i % 5, i % 2 == 0, &mut rng, &mut t);
            t
        })
        .collect();
    let mut fit = vec![0.0; cfg.population];
    let mut raw = vec![0.0; cfg.population];
    let mut history = Vec::with_capacity(cfg.generations + 1);
    let evaluate = |pop: &[Tree], fit: &mut [f64], raw: &mut [f64]| {
        let per = pop.len().div_ceil(threads.max(1));
        std::thread::scope(|s| {
            for ((ts, fs), rs) in pop
                .chunks(per)
                .zip(fit.chunks_mut(per))
                .zip(raw.chunks_mut(per))
            {
                s.spawn(move || {
                    let mut stack = Vec::with_capacity(64);
                    let mut z = Vec::with_capacity(train.len());
                    for ((t, f), r) in ts.iter().zip(fs.iter_mut()).zip(rs.iter_mut()) {
                        z.clear();
                        z.extend(train.x.iter().map(|x| eval(t, x, &mut stack)));
                        *r = fit_scaling(&z, &train.y).2;
                        *f = *r + cfg.parsimony * t.len() as f64;
                    }
                });
            }
        });
    };
    let mut child = Vec::new();
    for generation in 0..=cfg.generations {
        evaluate(&pop, &mut fit, &mut raw);
        let bi = (0..pop.len())
            .min_by(|&a, &b| fit[a].total_cmp(&fit[b]))
            .unwrap_or(0);
        let st = GpStats {
            generation,
            best_bce: raw.iter().copied().fold(f64::INFINITY, f64::min),
            mean_size: pop.iter().map(Vec::len).sum::<usize>() as f64 / pop.len() as f64,
            best_size: pop[bi].len(),
        };
        log(&st);
        history.push(st);
        if generation == cfg.generations {
            let mut stack = Vec::new();
            let z: Vec<f64> = train
                .x
                .iter()
                .map(|x| eval(&pop[bi], x, &mut stack))
                .collect();
            let (a, b, _) = fit_scaling(&z, &train.y);
            return GpResult {
                best: pop[bi].clone(),
                scaling: (a, b),
                history,
            };
        }
        let tour = |rng: &mut Rng| {
            let mut best = rng.index(pop.len());
            for _ in 1..cfg.tournament {
                let c = rng.index(pop.len());
                if fit[c] < fit[best] {
                    best = c;
                }
            }
            best
        };
        let mut order: Vec<usize> = (0..pop.len()).collect();
        order.sort_by(|&a, &b| fit[a].total_cmp(&fit[b]));
        let mut next: Vec<Tree> = order
            .iter()
            .take(cfg.elitism)
            .map(|&i| pop[i].clone())
            .collect();
        while next.len() < cfg.population {
            let a = tour(&mut rng);
            let r = rng.f64();
            if r < cfg.p_crossover {
                let b = tour(&mut rng);
                crossover(&pop[a], &pop[b], &mut rng, &mut child);
            } else if r < cfg.p_crossover + cfg.p_mutation {
                mutate(&pop[a], &mut rng, &mut child);
            } else {
                child.clone_from(&pop[a]);
            }
            // Límites duros (Koza): si el hijo excede, se copia el padre.
            if child.len() > cfg.max_size || depth(&child) > cfg.max_depth {
                child.clone_from(&pop[a]);
            }
            next.push(child.clone());
        }
        pop = next;
    }
    unreachable!("el bucle siempre termina en la última generación")
}

/// Escribe el árbol en notación infija legible.
pub fn to_infix(t: &[Node]) -> String {
    fn go(t: &[Node], i: usize, s: &mut String) -> usize {
        match t[i] {
            Node::Feat(f) => {
                s.push_str(FEATURE_NAMES[usize::from(f)]);
                i + 1
            }
            Node::Const(c) => {
                let _ = write!(s, "{c:.2}");
                i + 1
            }
            Node::IfGt => {
                s.push_str("si(");
                let j = go(t, i + 1, s);
                s.push_str(" > ");
                let k = go(t, j, s);
                s.push_str(", ");
                let l = go(t, k, s);
                s.push_str(", ");
                let m = go(t, l, s);
                s.push(')');
                m
            }
            Node::Min | Node::Max => {
                s.push_str(if t[i] == Node::Min { "min(" } else { "max(" });
                let j = go(t, i + 1, s);
                s.push_str(", ");
                let k = go(t, j, s);
                s.push(')');
                k
            }
            op => {
                s.push('(');
                let j = go(t, i + 1, s);
                s.push_str(match op {
                    Node::Add => " + ",
                    Node::Sub => " − ",
                    Node::Mul => " × ",
                    _ => " ÷ ",
                });
                let k = go(t, j, s);
                s.push(')');
                k
            }
        }
    }
    let mut s = String::new();
    if !t.is_empty() {
        go(t, 0, &mut s);
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluacion_y_subarboles() {
        // (x0 + 2) × x1
        let t = vec![
            Node::Mul,
            Node::Add,
            Node::Feat(0),
            Node::Const(2.0),
            Node::Feat(1),
        ];
        let mut x = [0.0; NF];
        x[0] = 3.0;
        x[1] = 4.0;
        let mut st = Vec::new();
        assert!((eval(&t, &x, &mut st) - 20.0).abs() < 1e-9);
        assert_eq!(subtree_end(&t, 0), 5);
        assert_eq!(subtree_end(&t, 1), 4);
        assert_eq!(depth(&t), 3);
        assert_eq!(to_infix(&t), "((huecos + 2.00) × hueco_max)");
        let div = vec![Node::Div, Node::Const(1.0), Node::Const(0.0)];
        assert!((eval(&div, &x, &mut st) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn operadores_producen_arboles_validos() {
        let mut rng = Rng::seed_from_u64(1);
        let mut out = Vec::new();
        for _ in 0..200 {
            let mut a = Vec::new();
            random_tree(4, false, &mut rng, &mut a);
            let mut b = Vec::new();
            random_tree(4, true, &mut rng, &mut b);
            crossover(&a, &b, &mut rng, &mut out);
            assert_eq!(subtree_end(&out, 0), out.len());
            mutate(&a, &mut rng, &mut out);
            assert_eq!(subtree_end(&out, 0), out.len());
        }
    }

    #[test]
    fn escalado_con_datos_separables() {
        let z: Vec<f64> = (0..600).map(|i| f64::from(i) / 60.0).collect();
        let y: Vec<f64> = z.iter().map(|&v| f64::from(u8::from(v <= 4.0))).collect();
        let (a, b, l) = fit_scaling(&z, &y);
        // La frontera de decisión −b/a debe quedar cerca del 4 real.
        assert!(a < 0.0 && (-b / a - 4.0).abs() < 0.3, "a={a} b={b} bce={l}");
    }

    #[test]
    fn aprende_una_regla_simple() {
        let mut rng = Rng::seed_from_u64(2);
        let mut d = Dataset::default();
        for _ in 0..600 {
            let mut x = [0.0; NF];
            x[1] = rng.f64() * 10.0;
            d.y.push(f64::from(u8::from(x[1] <= 4.0)));
            d.x.push(x);
        }
        let cfg = GpConfig {
            population: 200,
            generations: 20,
            ..GpConfig::default()
        };
        let r = evolve(&d, &cfg, 2, |_| {});
        let mut st = Vec::new();
        let acc =
            d.x.iter()
                .zip(&d.y)
                .filter(|(x, y)| (predict(&r.best, r.scaling, x, &mut st) >= 0.5) == (**y >= 0.5))
                .count();
        assert!(acc as f64 / 600.0 > 0.9, "exactitud {acc}/600");
    }
}
