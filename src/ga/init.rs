//! Inicialización de individuos y reparación.

use crate::level::{HEIGHT, Level, SAFE_COLS, Style, Tile};
use crate::rng::Rng;

/// Fila de la superficie del suelo en las zonas fijas de salida y meta.
pub const SAFE_GROUND: usize = HEIGHT - 2;

/// Escribe las zonas fijas: suelo, salida en la columna 1 y meta en la penúltima.
pub fn write_safe_zones(level: &mut Level) {
    let w = level.width();
    for x in (0..SAFE_COLS).chain(w - SAFE_COLS..w) {
        level.set_ground_column(x, SAFE_GROUND);
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
    }
    if full {
        // Quitar una plataforma puede dejar aislada a su vecina: hasta que no cambie.
        loop {
            let before = level.tiles().to_vec();
            repair_platforms(level);
            if level.tiles() == before.as_slice() {
                break;
            }
        }
        // Las escaleras dependen de las plataformas (salidas): van después.
        repair_ladders(level);
        // Al final, los objetos que necesitan apoyo sólido caen.
        for x in SAFE_COLS..w - SAFE_COLS {
            let col = level.column_mut(x);
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
        space_enemies(level);
    }
    write_safe_zones(level);
}

/// Separación mínima entre dos enemigos de la misma fila. Cada uno patrulla una
/// casilla a cada lado, así que con 5 columnas quedan 2 libres entre sus zonas
/// para aterrizar entre ellos.
pub const ENEMY_SPACING: usize = 5;

/// Quita los enemigos demasiado pegados a otro de su fila (de izquierda a
/// derecha se conserva el primero): las filas compactas cierran el paso.
fn space_enemies(level: &mut Level) {
    let mut last = [None::<usize>; HEIGHT];
    for x in SAFE_COLS..level.width() - SAFE_COLS {
        for (y, prev) in last.iter_mut().enumerate() {
            if level.get(x, y) != Tile::Enemy {
                continue;
            }
            if prev.is_some_and(|p| x - p < ENEMY_SPACING) {
                level.set(x, y, Tile::Empty);
            } else {
                *prev = Some(x);
            }
        }
    }
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
            let mut new_top = top;
            if supported {
                while new_top <= bottom && !level.ladder_exit(xi, new_top as i32) {
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

/// Repara las plataformas flotantes según lo observado en niveles hechos a
/// mano: no hay plataformas aisladas de un tile y bajo cada plataforma quedan al
/// menos 2 filas libres antes del suelo (si no, bloquean el paso o el salto).
pub fn repair_platforms(level: &mut Level) {
    let w = level.width();
    for x in SAFE_COLS..w - SAFE_COLS {
        for y in 1..HEIGHT {
            if level.get(x, y) != Tile::Platform {
                continue;
            }
            let (xi, yi) = (x as i32, y as i32);
            let isolated = level.get_or_empty(xi - 1, yi) != Tile::Platform
                && level.get_or_empty(xi + 1, yi) != Tile::Platform;
            let cramped = (1..=2).any(|d| level.get_or_empty(xi, yi + d) == Tile::Ground);
            if isolated || cramped {
                level.set(x, y, Tile::Empty);
            }
        }
    }
}

/// Genera un nivel con estructura de diseño a partir de la biblioteca de
/// patrones ([`crate::patterns`]): se elige un tema (mecánica A y mecánica B) y
/// se encadenan grupos de ritmo de 1 a 3 patrones separados por descansos,
/// siguiendo cuatro actos (presentar, desarrollar, giro y conclusión) con la
/// dificultad subiendo hasta el giro y bajando al final. Es la distribución
/// inicial del GA y de la búsqueda aleatoria (para que la comparación sea justa).
pub fn random_level(width: usize, rng: &mut Rng) -> Level {
    use crate::patterns::{self, Kind};
    let style = Style::ALL[rng.index(Style::COUNT)];
    let mut l = Level::empty(width, style);
    write_safe_zones(&mut l);
    let theme = patterns::THEMES[rng.index(patterns::THEMES.len())];
    let end = width - SAFE_COLS;
    let mut x = SAFE_COLS;
    let mut g = SAFE_GROUND;
    // Descanso inicial.
    let b = patterns::build(&mut l, Kind::Rest, x, g, 0.0, rng);
    x += b.width;
    while x < end {
        let progress = (x - SAFE_COLS) as f64 / (end - SAFE_COLS) as f64;
        let act = ((progress * 4.0) as usize).min(3);
        let difficulty = [0.1, 0.4, 0.8, 0.3][act];
        let group = 1 + rng.index(3);
        for _ in 0..group {
            let kind = patterns::pick(rng, theme, act);
            if x + patterns::max_width(kind) + 3 > end {
                break;
            }
            let b = patterns::build(&mut l, kind, x, g, difficulty, rng);
            x += b.width;
            g = b.exit;
        }
        // Descanso entre grupos (o relleno final si no cabe nada más).
        if x + patterns::max_width(Kind::Rest) > end {
            for cx in x..end {
                l.set_ground_column(cx, g);
            }
            break;
        }
        let b = patterns::build(&mut l, Kind::Rest, x, g, 0.0, rng);
        x += b.width;
    }
    patterns::smooth_junction(&mut l, end - 1);
    repair(&mut l, true);
    l
}

/// Genera un nivel de ruido: cada celda recibe un tile evolucionable al azar
/// y solo se escriben las zonas fijas de salida y meta. No tiene estructura
/// ni reparación; es el punto de partida del modo evolución del juego.
pub fn noise_level(width: usize, rng: &mut Rng) -> Level {
    let mut l = Level::empty(width, Style::ALL[rng.index(Style::COUNT)]);
    for x in SAFE_COLS..width - SAFE_COLS {
        for y in 0..HEIGHT {
            l.set(x, y, Tile::EVOLVABLE[rng.index(Tile::EVOLVABLE.len())]);
        }
    }
    write_safe_zones(&mut l);
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

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn zonas_fijas_y_unica_salida_meta() {
        let mut rng = Rng::seed_from_u64(1);
        for k in 0..50 {
            let l = if k % 2 == 0 {
                random_level(80, &mut rng)
            } else {
                noise_level(80, &mut rng)
            };
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

    /// Nivel de ruido: cada celda uniforme entre los tiles evolucionables.
    pub(crate) fn noise_level(width: usize, rng: &mut Rng) -> Level {
        let style = Style::ALL[rng.index(Style::COUNT)];
        let mut l = Level::empty(width, style);
        for t in l.tiles_mut() {
            *t = Tile::EVOLVABLE[rng.index(Tile::EVOLVABLE.len())];
        }
        repair(&mut l, false);
        l
    }

    #[test]
    fn reparacion_idempotente() {
        let mut rng = Rng::seed_from_u64(2);
        let mut l = noise_level(60, &mut rng);
        repair(&mut l, true);
        let once = l.clone();
        repair(&mut l, true);
        assert_eq!(l, once);
    }
}
