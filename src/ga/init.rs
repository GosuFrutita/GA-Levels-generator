//! Inicialización de individuos y reparación.

use crate::level::{HEIGHT, Level, SAFE_COLS, Style, Tile};
use crate::rng::Rng;

/// Fila de la superficie del suelo en las zonas fijas de salida y meta.
pub const SAFE_GROUND: usize = HEIGHT - 2;

/// Escribe las zonas fijas: suelo, salida en la columna 1 y meta en la penúltima.
pub fn write_safe_zones(level: &mut Level) {
    let w = level.width();
    for x in (0..SAFE_COLS).chain(w - SAFE_COLS..w) {
        let col = level.column_mut(x);
        col.fill(Tile::Empty);
        col[SAFE_GROUND..].fill(Tile::Ground);
    }
    level.set(1, SAFE_GROUND - 1, Tile::Start);
    level.set(w - 2, SAFE_GROUND - 1, Tile::Goal);
}

/// Reparación ligera y determinista.
///
/// Siempre: restaura las zonas fijas y elimina `Start`/`Goal` sueltos. Si
/// `full`: convierte el suelo flotante en plataforma y deja caer enemigos,
/// peligros y decoraciones hasta el primer apoyo sólido (o los elimina si caen
/// a un foso) y repara las escaleras ([`repair_ladders`]). La variante sin
/// reparación completa se compara en un experimento.
pub fn repair(level: &mut Level, full: bool) {
    let w = level.width();
    for x in SAFE_COLS..w - SAFE_COLS {
        let col = level.column_mut(x);
        for t in col.iter_mut() {
            if matches!(*t, Tile::Start | Tile::Goal) {
                *t = Tile::Empty;
            }
        }
        if !full {
            continue;
        }
        // De abajo arriba: el suelo solo es suelo si apoya en suelo (o en el fondo).
        for y in (0..HEIGHT - 1).rev() {
            if col[y] == Tile::Ground && col[y + 1] != Tile::Ground {
                col[y] = Tile::Platform;
            }
        }
        // Objetos que necesitan apoyo sólido: caen.
        for y in (0..HEIGHT).rev() {
            let t = col[y];
            if !matches!(t, Tile::Enemy | Tile::Hazard | Tile::Decor) {
                continue;
            }
            let mut ny = y;
            while ny + 1 < HEIGHT && col[ny + 1] == Tile::Empty {
                ny += 1;
            }
            let supported = ny + 1 < HEIGHT && col[ny + 1].is_solid();
            col[y] = Tile::Empty;
            if supported {
                col[ny] = t;
            }
        }
    }
    if full {
        repair_ladders(level);
    }
    write_safe_zones(level);
}

/// ¿Se puede estar de pie en `(x, y)`?
fn can_stand(level: &Level, x: i32, y: i32) -> bool {
    y >= 0 && level.get_or_empty(x, y).is_passable() && level.get_or_empty(x, y + 1).is_support()
}

/// Repara las escaleras: elimina los tramos que no apoyan en algo sólido,
/// recorta por arriba la parte que no lleva a ninguna salida y elimina los
/// tramos que quedan con menos de 2 tiles. Sin esto el GA usaba escaleras como
/// pilares en mitad de los huecos y como relleno para sumar variedad.
pub fn repair_ladders(level: &mut Level) {
    let w = level.width();
    for x in SAFE_COLS..w - SAFE_COLS {
        let mut y = 0;
        while y < HEIGHT {
            if level.get(x, y) != Tile::Ladder {
                y += 1;
                continue;
            }
            let top = y;
            while y < HEIGHT && level.get(x, y) == Tile::Ladder {
                y += 1;
            }
            let bottom = y - 1;
            let xi = x as i32;
            let supported = y < HEIGHT && level.get(x, y).is_solid();
            let exit = |l: &Level, t: usize| {
                let ti = t as i32;
                can_stand(l, xi - 1, ti)
                    || can_stand(l, xi + 1, ti)
                    || can_stand(l, xi - 1, ti - 1)
                    || can_stand(l, xi + 1, ti - 1)
            };
            let mut new_top = top;
            if supported {
                while new_top <= bottom && !exit(level, new_top) {
                    new_top += 1;
                }
            } else {
                new_top = bottom + 1;
            }
            if bottom + 1 < new_top + 2 {
                new_top = bottom + 1; // queda corta: se elimina entera
            }
            for yy in top..new_top {
                level.set(x, yy, Tile::Empty);
            }
        }
    }
}

/// Genera un nivel aleatorio "plausible" con un generador constructivo:
/// superficie por paseo aleatorio con huecos, plataformas, enemigos, peligros,
/// monedas, escaleras y decoración. Es la distribución inicial del GA y de la
/// búsqueda aleatoria (para que la comparación sea justa).
pub fn random_level(width: usize, rng: &mut Rng) -> Level {
    let style = Style::ALL[rng.index(Style::COUNT)];
    let mut l = Level::empty(width, style);
    let lo = HEIGHT - 6; // superficie más alta permitida al iniciar
    let mut surface = SAFE_GROUND;
    let mut x = SAFE_COLS;
    let end = width - SAFE_COLS;
    while x < end {
        if rng.chance(0.07) {
            // Hueco de 1 a 4 columnas.
            let gap = 1 + rng.index(4);
            x += gap.min(end - x);
            continue;
        }
        if rng.chance(0.15) {
            let step = rng.range_i32(-2, 2);
            surface = (surface as i32 + step).clamp(lo as i32, HEIGHT as i32 - 1) as usize;
        }
        l.column_mut(x)[surface..].fill(Tile::Ground);
        x += 1;
    }
    let span = end - SAFE_COLS;
    let count = |per: f64, rng: &mut Rng| -> usize {
        let mean = span as f64 * per;
        (mean * (0.5 + rng.f64())).round() as usize
    };
    // Plataformas flotantes.
    for _ in 0..count(0.08, rng) {
        let len = 2 + rng.index(5);
        let x0 = SAFE_COLS + rng.index(span.saturating_sub(len).max(1));
        let y = 3 + rng.index(HEIGHT - 6);
        for xx in x0..(x0 + len).min(end) {
            if l.get(xx, y) == Tile::Empty {
                l.set(xx, y, Tile::Platform);
            }
        }
    }
    // Objetos colocados sobre superficies.
    let place_on_surface = |tile: Tile, n: usize, rng: &mut Rng, l: &mut Level| {
        for _ in 0..n {
            let x = SAFE_COLS + rng.index(span);
            let col = l.column(x);
            // Superficies disponibles en la columna: celda vacía con sólido debajo.
            let tops: Vec<usize> = (0..HEIGHT - 1)
                .filter(|&y| col[y] == Tile::Empty && col[y + 1].is_solid())
                .collect();
            if !tops.is_empty() {
                let y = tops[rng.index(tops.len())];
                l.set(x, y, tile);
            }
        }
    };
    place_on_surface(Tile::Enemy, count(0.05, rng), rng, &mut l);
    place_on_surface(Tile::Hazard, count(0.03, rng), rng, &mut l);
    place_on_surface(Tile::Decor, count(0.08, rng), rng, &mut l);
    // Monedas: en el aire, 1-2 celdas sobre una superficie.
    for _ in 0..count(0.12, rng) {
        let x = SAFE_COLS + rng.index(span);
        let y = 1 + rng.index(HEIGHT - 3);
        if l.get(x, y) == Tile::Empty {
            l.set(x, y, Tile::Coin);
        }
    }
    // Escaleras: tramos verticales que bajan hasta un apoyo.
    for _ in 0..count(0.015, rng) {
        let x = SAFE_COLS + rng.index(span);
        let top = 2 + rng.index(HEIGHT / 2);
        let mut y = top;
        while y < HEIGHT && l.get(x, y) == Tile::Empty {
            l.set(x, y, Tile::Ladder);
            y += 1;
        }
    }
    repair(&mut l, true);
    l
}

/// Recorta la ventana de columnas `[x0, x0 + width)` de un nivel y la
/// convierte en un nivel completo (con sus zonas de salida y meta).
///
/// # Panics
///
/// Si la ventana se sale del nivel o es demasiado estrecha.
pub fn window(level: &Level, x0: usize, width: usize) -> Level {
    assert!(x0 + width <= level.width(), "ventana fuera del nivel");
    let tiles = level.tiles()[x0 * HEIGHT..(x0 + width) * HEIGHT].to_vec();
    let mut out = Level::from_tiles(tiles, level.style).expect("ancho suficiente");
    repair(&mut out, true);
    out
}

/// Nivel completamente aleatorio (cada celda uniforme entre los tiles
/// evolucionables). Línea base "sin conocimiento del dominio".
pub fn uniform_level(width: usize, rng: &mut Rng) -> Level {
    let style = Style::ALL[rng.index(Style::COUNT)];
    let mut l = Level::empty(width, style);
    for t in l.tiles_mut() {
        *t = Tile::EVOLVABLE[rng.index(Tile::EVOLVABLE.len())];
    }
    repair(&mut l, false);
    l
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zonas_fijas_y_unica_salida_meta() {
        let mut rng = Rng::seed_from_u64(1);
        for _ in 0..50 {
            let l = random_level(80, &mut rng);
            assert_eq!(l.tiles().iter().filter(|&&t| t == Tile::Start).count(), 1);
            assert_eq!(l.tiles().iter().filter(|&&t| t == Tile::Goal).count(), 1);
            assert_eq!(l.start(), Some((1, SAFE_GROUND - 1)));
        }
    }

    #[test]
    fn reparacion_de_escaleras() {
        let mut l = Level::empty(30, Style::Base);
        write_safe_zones(&mut l);
        for x in 0..30 {
            l.set(x, HEIGHT - 1, Tile::Ground);
        }
        // Escalera de 6 sin nada arriba: se elimina entera.
        for y in HEIGHT - 7..HEIGHT - 1 {
            l.set(10, y, Tile::Ladder);
        }
        // Escalera de 6 con plataforma a media altura: se recorta hasta ahí.
        for y in HEIGHT - 7..HEIGHT - 1 {
            l.set(20, y, Tile::Ladder);
        }
        l.set(21, HEIGHT - 3, Tile::Platform);
        l.set(22, HEIGHT - 3, Tile::Platform);
        // Escalera colgando sobre un hueco: se elimina.
        l.set(15, HEIGHT - 1, Tile::Empty);
        l.set(15, HEIGHT - 2, Tile::Ladder);
        l.set(15, HEIGHT - 3, Tile::Ladder);
        repair_ladders(&mut l);
        assert!((0..HEIGHT).all(|y| l.get(10, y) != Tile::Ladder));
        assert!((0..HEIGHT).all(|y| l.get(15, y) != Tile::Ladder));
        let kept: Vec<usize> = (0..HEIGHT)
            .filter(|&y| l.get(20, y) == Tile::Ladder)
            .collect();
        assert_eq!(kept.first(), Some(&(HEIGHT - 4)), "{kept:?}");
        assert!(kept.len() >= 2);
    }

    #[test]
    fn reparacion_idempotente() {
        let mut rng = Rng::seed_from_u64(2);
        let mut l = uniform_level(60, &mut rng);
        repair(&mut l, true);
        let once = l.clone();
        repair(&mut l, true);
        assert_eq!(l, once);
    }
}
