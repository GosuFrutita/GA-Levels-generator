//! Operadores de variación: cruce y mutación sobre la matriz por columnas.

use crate::level::{HEIGHT, Level, SAFE_COLS, Style, Tile};
use crate::rng::Rng;

/// Operador de cruce.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crossover {
    /// Dos puntos por columnas, prefiriendo cortes donde ambos padres tienen el
    /// suelo a la misma altura ("costuras" limpias).
    Columns,
    /// Intercambio de un rectángulo 2D.
    Block,
    /// Uniforme celda a celda (línea base destructiva).
    Uniform,
    /// Sin cruce (los hijos son copias; solo muta).
    None,
}

impl Crossover {
    /// Nombre para informes.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Columns => "columnas",
            Self::Block => "bloque",
            Self::Uniform => "uniforme",
            Self::None => "ninguno",
        }
    }
}

/// Elige una columna de corte en `[lo, hi)`, prefiriendo costuras limpias.
fn cut(a: &Level, b: &Level, lo: usize, hi: usize, rng: &mut Rng) -> usize {
    let mut fallback = lo + rng.index(hi - lo);
    for _ in 0..8 {
        let c = lo + rng.index(hi - lo);
        if a.ground_top(c - 1) == b.ground_top(c - 1) && a.ground_top(c) == b.ground_top(c) {
            return c;
        }
        fallback = c;
    }
    fallback
}

/// Cruza `a` y `b` escribiendo los hijos en `ca` y `cb` (que ya deben ser copias
/// de `a` y `b`, respectivamente). Devuelve el número de columnas intercambiadas.
pub fn crossover(
    op: Crossover,
    a: &Level,
    b: &Level,
    ca: &mut Level,
    cb: &mut Level,
    rng: &mut Rng,
) -> usize {
    let w = a.width();
    debug_assert_eq!(w, b.width());
    let (lo, hi) = (SAFE_COLS, w - SAFE_COLS);
    match op {
        Crossover::None => 0,
        Crossover::Columns => {
            let mut c1 = cut(a, b, lo + 1, hi, rng);
            let mut c2 = cut(a, b, lo + 1, hi, rng);
            if c1 > c2 {
                std::mem::swap(&mut c1, &mut c2);
            }
            let r = c1 * HEIGHT..c2 * HEIGHT;
            ca.tiles_mut()[r.clone()].copy_from_slice(&b.tiles()[r.clone()]);
            cb.tiles_mut()[r.clone()].copy_from_slice(&a.tiles()[r]);
            // El gen de estilo se hereda del padre que aporta más columnas.
            if 2 * (c2 - c1) > w {
                ca.style = b.style;
                cb.style = a.style;
            }
            c2 - c1
        }
        Crossover::Block => {
            let bw = 4 + rng.index((w / 3).max(5) - 4);
            let x0 = lo + rng.index((hi - lo).saturating_sub(bw).max(1));
            let x1 = (x0 + bw).min(hi);
            let bh = 3 + rng.index(HEIGHT - 2);
            let y0 = rng.index(HEIGHT - bh.min(HEIGHT) + 1);
            let y1 = (y0 + bh).min(HEIGHT);
            for x in x0..x1 {
                let r = x * HEIGHT + y0..x * HEIGHT + y1;
                ca.tiles_mut()[r.clone()].copy_from_slice(&b.tiles()[r.clone()]);
                cb.tiles_mut()[r.clone()].copy_from_slice(&a.tiles()[r]);
            }
            if rng.chance(0.5) {
                std::mem::swap(&mut ca.style, &mut cb.style);
            }
            x1 - x0
        }
        Crossover::Uniform => {
            let r = lo * HEIGHT..hi * HEIGHT;
            for i in r {
                if rng.chance(0.5) {
                    ca.tiles_mut()[i] = b.tiles()[i];
                    cb.tiles_mut()[i] = a.tiles()[i];
                }
            }
            if rng.chance(0.5) {
                std::mem::swap(&mut ca.style, &mut cb.style);
            }
            (hi - lo) / 2
        }
    }
}

/// Parámetros de mutación.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mutation {
    /// Probabilidad de mutar cada celda evolutiva (p. ej. `3/L`).
    pub per_tile: f64,
    /// Probabilidad de aplicar una mutación "macro" (motivo) por hijo.
    pub macro_rate: f64,
    /// Probabilidad de cambiar el gen de estilo por hijo.
    pub style_rate: f64,
}

/// Mutaciones macro: insertan motivos estructurales completos.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MacroOp {
    /// Abrir un hueco de 1 a 4 columnas.
    Gap,
    /// Subir o bajar el suelo en un tramo de columnas.
    Shift,
    /// Plataforma flotante de 2 a 5 tiles.
    Platform,
    /// Enemigo sobre una superficie.
    Enemy,
    /// Tramo de escalera.
    Ladder,
    /// Copiar un tramo de columnas a otra posición (análogo a la mutación
    /// convolucional de Lucas y Volz, 2019).
    CopyBlock,
    /// Fila de monedas sobre una superficie.
    Coins,
}

impl MacroOp {
    /// Todos los operadores macro.
    pub const ALL: [Self; 7] = [
        Self::Gap,
        Self::Shift,
        Self::Platform,
        Self::Enemy,
        Self::Ladder,
        Self::CopyBlock,
        Self::Coins,
    ];
}

/// Mutación puntual: cada celda de la región evolutiva cambia con probabilidad
/// `p` a un tile evolucionable distinto. Usa saltos geométricos para no tirar
/// un número aleatorio por celda cuando `p` es pequeño.
pub fn point_mutation(level: &mut Level, p: f64, rng: &mut Rng) -> usize {
    if p <= 0.0 {
        return 0;
    }
    let w = level.width();
    let r = SAFE_COLS * HEIGHT..(w - SAFE_COLS) * HEIGHT;
    let ln_q = (1.0 - p.min(0.999_999)).ln();
    let mut i = r.start;
    let mut n = 0;
    loop {
        // Salto ~ Geométrica(p): número de celdas que no mutan.
        let u = 1.0 - rng.f64();
        i += (u.ln() / ln_q).floor() as usize;
        if i >= r.end {
            return n;
        }
        let old = level.tiles()[i];
        let mut t = Tile::EVOLVABLE[rng.index(Tile::EVOLVABLE.len() - 1)];
        if t == old {
            t = Tile::EVOLVABLE[Tile::EVOLVABLE.len() - 1];
        }
        level.tiles_mut()[i] = t;
        n += 1;
        i += 1;
    }
}

/// Aplica una mutación macro concreta.
pub fn macro_mutation(level: &mut Level, op: MacroOp, rng: &mut Rng) {
    let w = level.width();
    let (lo, hi) = (SAFE_COLS, w - SAFE_COLS);
    let x = lo + rng.index(hi - lo);
    let surface = |l: &Level, x: usize| {
        (0..HEIGHT - 1)
            .rev()
            .find(|&y| l.get(x, y) == Tile::Empty && l.get(x, y + 1).is_solid())
    };
    match op {
        MacroOp::Gap => {
            for xx in x..(x + 1 + rng.index(4)).min(hi) {
                let col = level.column_mut(xx);
                for t in col.iter_mut().filter(|t| **t == Tile::Ground) {
                    *t = Tile::Empty;
                }
            }
        }
        MacroOp::Shift => {
            let len = 2 + rng.index(8);
            let up = rng.chance(0.5);
            for xx in x..(x + len).min(hi) {
                let top = level.ground_top(xx);
                let col = level.column_mut(xx);
                if up && top > HEIGHT - 7 && top <= HEIGHT {
                    if top < HEIGHT {
                        col[top - 1] = Tile::Ground;
                    }
                } else if !up && top < HEIGHT - 1 {
                    col[top] = Tile::Empty;
                }
            }
        }
        MacroOp::Platform => {
            let y = 3 + rng.index(HEIGHT - 6);
            for xx in x..(x + 2 + rng.index(4)).min(hi) {
                if level.get(xx, y) == Tile::Empty {
                    level.set(xx, y, Tile::Platform);
                }
            }
        }
        MacroOp::Enemy => {
            if let Some(y) = surface(level, x) {
                level.set(x, y, Tile::Enemy);
            }
        }
        MacroOp::Ladder => {
            if let Some(base) = surface(level, x) {
                let len = 2 + rng.index(4);
                for y in base.saturating_sub(len - 1)..=base {
                    if level.get(x, y) == Tile::Empty {
                        level.set(x, y, Tile::Ladder);
                    }
                }
            }
        }
        MacroOp::CopyBlock => {
            let len = 3 + rng.index(6);
            let src = lo + rng.index((hi - lo).saturating_sub(len).max(1));
            let dst = lo + rng.index((hi - lo).saturating_sub(len).max(1));
            let len = len.min(hi - src).min(hi - dst);
            let tiles = level.tiles_mut();
            tiles.copy_within(src * HEIGHT..(src + len) * HEIGHT, dst * HEIGHT);
        }
        MacroOp::Coins => {
            if let Some(y) = surface(level, x) {
                let cy = y.saturating_sub(1 + rng.index(2));
                for xx in x..(x + 2 + rng.index(3)).min(hi) {
                    if level.get(xx, cy) == Tile::Empty {
                        level.set(xx, cy, Tile::Coin);
                    }
                }
            }
        }
    }
}

/// Cambia el estilo a uno de los otros tres, uniformemente.
pub fn style_mutation(level: &mut Level, rng: &mut Rng) {
    let cur = level.style.index();
    let mut s = rng.index(Style::COUNT - 1);
    if s >= cur {
        s += 1;
    }
    level.style = Style::ALL[s];
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ga::init::random_level;

    #[test]
    fn cruce_por_columnas_conserva_el_material() {
        let mut rng = Rng::seed_from_u64(3);
        let a = random_level(64, &mut rng);
        let b = random_level(64, &mut rng);
        let (mut ca, mut cb) = (a.clone(), b.clone());
        crossover(Crossover::Columns, &a, &b, &mut ca, &mut cb, &mut rng);
        // Cada columna de los hijos viene de un padre y la del otro hijo del otro.
        for x in 0..64 {
            let from_a = ca.column(x) == a.column(x) && cb.column(x) == b.column(x);
            let from_b = ca.column(x) == b.column(x) && cb.column(x) == a.column(x);
            assert!(from_a || from_b, "columna {x}");
        }
    }

    #[test]
    fn mutacion_puntual_tasa_esperada() {
        let mut rng = Rng::seed_from_u64(4);
        let mut l = random_level(100, &mut rng);
        let cells = (100 - 2 * SAFE_COLS) * HEIGHT;
        let mut total = 0;
        for _ in 0..2000 {
            total += point_mutation(&mut l, 3.0 / cells as f64, &mut rng);
        }
        let mean = total as f64 / 2000.0;
        assert!((mean - 3.0).abs() < 0.2, "media {mean}");
    }

    #[test]
    fn mutacion_de_estilo_siempre_cambia() {
        let mut rng = Rng::seed_from_u64(5);
        let mut l = random_level(40, &mut rng);
        for _ in 0..100 {
            let before = l.style;
            style_mutation(&mut l, &mut rng);
            assert_ne!(before, l.style);
        }
    }
}
