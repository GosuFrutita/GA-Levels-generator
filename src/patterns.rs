//! Patrones de diseño de niveles de plataformas: constructores y detector.
//!
//! Fundamento (ver `research_notes/.../diseno_niveles_plataformas.md` y
//! `repos_godot_plataformas.md`):
//! - Los niveles diseñados se componen de *patrones* reconocibles (Dahlskog y
//!   Togelius, 2012: huecos, valles, escalones, hordas, caminos múltiples...)
//!   agrupados en *grupos de ritmo* separados por zonas de descanso (Smith et
//!   al., 2008 y 2009). Colocar plataformas al azar "no es divertido" (Smith et
//!   al., 2009) y produce un amasijo caótico (Mawhorter y Mateas, 2010).
//! - En los niveles hechos a mano analizados (demos de Godot y plantillas) las
//!   plataformas flotantes miden 2 a 5 de ancho (moda 3), forman cadenas que
//!   suben 1 a 2 filas por paso, dejan 2 filas libres debajo y nunca están
//!   aisladas; el suelo cambia de altura cada 3 a 12 columnas y todo desnivel
//!   mayor que el salto tiene un conector.
//!
//! La geometría usa nuestra envolvente de salto: 3 filas de alto y 4 columnas
//! de largo. El cromosoma sigue siendo la matriz de tiles: los patrones solo
//! *escriben* bloques en ella (inicialización y mutación) y el detector solo
//! *lee* la matriz (aptitud).

use crate::level::{HEIGHT, Level, SAFE_COLS, Tile};
use crate::rng::Rng;

/// Fila más alta permitida para la superficie del suelo.
pub const MIN_GROUND: usize = 8;
/// Fila más baja de la superficie del suelo (una sola fila de tierra).
pub const MAX_GROUND: usize = HEIGHT - 1;

/// Tipo de patrón.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Kind {
    /// Tramo llano de descanso.
    Rest,
    /// Hueco simple.
    Gap,
    /// Varios huecos con islas.
    MultiGap,
    /// Escalones de subida o bajada.
    Stairs,
    /// Meseta: subida, llano elevado y bajada.
    Mesa,
    /// Zanja con enemigo dentro.
    Trench,
    /// Varios enemigos en un tramo llano.
    Horde,
    /// Enemigo aislado.
    Enemy,
    /// Tira de pinchos.
    Spikes,
    /// Hueco largo cruzado por una plataforma flotante.
    Bridge,
    /// Cadena de plataformas que sube.
    Chain,
    /// Doble ruta: plataforma alta con recompensa sobre un camino con peligro.
    TwoPath,
    /// Huecos separados por pilares.
    PillarGap,
    /// Desnivel mayor que el salto con escalera de mano.
    LadderShaft,
}

impl Kind {
    /// Todos los patrones de la biblioteca.
    pub const ALL: [Self; 14] = [
        Self::Rest,
        Self::Gap,
        Self::MultiGap,
        Self::Stairs,
        Self::Mesa,
        Self::Trench,
        Self::Horde,
        Self::Enemy,
        Self::Spikes,
        Self::Bridge,
        Self::Chain,
        Self::TwoPath,
        Self::PillarGap,
        Self::LadderShaft,
    ];

    /// Categoría para la mezcla objetivo.
    pub const fn category(self) -> Category {
        match self {
            Self::Rest => Category::Rest,
            Self::Gap | Self::MultiGap | Self::PillarGap => Category::Gap,
            Self::Stairs | Self::Mesa => Category::Terrain,
            Self::Trench | Self::Horde | Self::Enemy => Category::Enemy,
            Self::Spikes => Category::Hazard,
            Self::Bridge | Self::Chain | Self::TwoPath => Category::Platform,
            Self::LadderShaft => Category::Ladder,
        }
    }

    /// Nombre para informes.
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rest => "descanso",
            Self::Gap => "hueco",
            Self::MultiGap => "multihueco",
            Self::Stairs => "escalones",
            Self::Mesa => "meseta",
            Self::Trench => "zanja",
            Self::Horde => "horda",
            Self::Enemy => "enemigo",
            Self::Spikes => "pinchos",
            Self::Bridge => "puente",
            Self::Chain => "cadena",
            Self::TwoPath => "doble ruta",
            Self::PillarGap => "pilares",
            Self::LadderShaft => "escalera",
        }
    }
}

/// Categoría de patrón.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Category {
    /// Descanso.
    Rest,
    /// Huecos.
    Gap,
    /// Relieve del terreno.
    Terrain,
    /// Enemigos.
    Enemy,
    /// Peligros fijos.
    Hazard,
    /// Plataformas flotantes.
    Platform,
    /// Escaleras de mano.
    Ladder,
}

impl Category {
    /// Categorías que cuentan para la mezcla (sin el descanso).
    pub const MIX: [Self; 6] = [
        Self::Gap,
        Self::Terrain,
        Self::Enemy,
        Self::Hazard,
        Self::Platform,
        Self::Ladder,
    ];

    /// Proporción objetivo en un nivel variado. La de plataformas es la mayor
    /// (≥ 30 %, contra los niveles "planos"); las fuentes no dan cifras exactas,
    /// así que es una propuesta calibrable.
    pub const fn target(self) -> f64 {
        match self {
            Self::Rest => 0.0,
            Self::Gap => 0.20,
            Self::Terrain => 0.17,
            Self::Enemy => 0.18,
            Self::Hazard => 0.10,
            Self::Platform => 0.30,
            Self::Ladder => 0.05,
        }
    }
}

// --- Escritura ---------------------------------------------------------------

fn set(level: &mut Level, x: usize, y: usize, t: Tile) {
    if x < level.width() && y < HEIGHT && y > 0 {
        level.set(x, y, t);
    }
}

/// Plataforma horizontal de un tile de grosor.
fn platform(level: &mut Level, x0: usize, width: usize, row: usize) {
    for x in x0..x0 + width {
        set(level, x, row, Tile::Platform);
    }
}

/// Fila de monedas.
fn coins(level: &mut Level, x0: usize, n: usize, row: usize) {
    for x in x0..x0 + n {
        if x < level.width() && row > 0 && level.get(x, row) == Tile::Empty {
            level.set(x, row, Tile::Coin);
        }
    }
}

/// Decoración sobre la superficie con probabilidad `p` por columna.
fn decorate(level: &mut Level, x0: usize, w: usize, g: usize, p: f64, rng: &mut Rng) {
    if g == 0 || g >= HEIGHT {
        return;
    }
    for x in x0..x0 + w {
        if x < level.width() && level.get(x, g - 1) == Tile::Empty && rng.chance(p) {
            level.set(x, g - 1, Tile::Decor);
        }
    }
}

/// Resultado de construir un patrón.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Built {
    /// Columnas ocupadas.
    pub width: usize,
    /// Superficie del suelo a la salida.
    pub exit: usize,
}

/// Anchura que ocupará un patrón (para saber si cabe).
pub const fn max_width(kind: Kind) -> usize {
    match kind {
        Kind::Rest | Kind::Enemy => 6,
        Kind::Gap | Kind::Spikes => 8,
        // La escalera de mano cae a escalones si no hay altura para ella.
        Kind::Mesa | Kind::Stairs | Kind::LadderShaft => 12,
        Kind::PillarGap => 18,
        Kind::Trench => 9,
        Kind::Bridge | Kind::TwoPath | Kind::MultiGap => 13,
        Kind::Chain | Kind::Horde => 16,
    }
}

/// Construye un patrón empezando en la columna `x0` con la superficie del suelo
/// en `g`. `difficulty ∈ [0, 1]` endurece los parámetros (huecos más anchos,
/// más enemigos, peligros bajo las rutas). El patrón nunca ocupa más de
/// [`max_width`] columnas.
pub fn build(
    level: &mut Level,
    kind: Kind,
    x0: usize,
    g: usize,
    difficulty: f64,
    rng: &mut Rng,
) -> Built {
    match kind {
        Kind::Rest => build_rest(level, x0, g, difficulty, rng),
        Kind::Gap => build_gap(level, x0, g, difficulty, rng),
        Kind::MultiGap => build_multi_gap(level, x0, g, difficulty, rng),
        Kind::Stairs => build_stairs(level, x0, g, difficulty, rng),
        Kind::Mesa => build_mesa(level, x0, g, difficulty, rng),
        Kind::Trench => build_trench(level, x0, g, difficulty, rng),
        Kind::Horde | Kind::Enemy => build_horde(level, kind, x0, g, difficulty, rng),
        Kind::Spikes => build_spikes(level, x0, g, difficulty, rng),
        Kind::Bridge => build_bridge(level, x0, g, difficulty, rng),
        Kind::Chain => build_chain(level, x0, g, difficulty, rng),
        Kind::TwoPath => build_two_path(level, x0, g, difficulty, rng),
        Kind::PillarGap => build_pillar_gap(level, x0, g, difficulty, rng),
        Kind::LadderShaft => build_ladder_shaft(level, x0, g, difficulty, rng),
    }
}

/// ¿Se aplica la variante difícil? Probabilidad igual a la dificultad.
fn hard(rng: &mut Rng, difficulty: f64) -> bool {
    rng.chance(difficulty)
}

/// Tramo llano de suelo a la altura `g` con algo de decoración.
fn flat(level: &mut Level, x: usize, w: usize, g: usize, rng: &mut Rng) {
    for i in 0..w {
        level.set_ground_column(x + i, g);
    }
    decorate(level, x, w, g, 0.35, rng);
}

/// Descanso: tramo llano con decoración y a veces una fila de monedas.
fn build_rest(level: &mut Level, x0: usize, g: usize, _difficulty: f64, rng: &mut Rng) -> Built {
    let w = 3 + rng.index(4);
    flat(level, x0, w, g, rng);
    if w >= 5 && rng.chance(0.4) {
        coins(level, x0 + 1, 3, g - 2);
    }
    Built { width: w, exit: g }
}

/// Hueco de 2 o 3 columnas con arco de monedas.
fn build_gap(level: &mut Level, x0: usize, g: usize, difficulty: f64, rng: &mut Rng) -> Built {
    // Hueco de 2 o 3: el salto en llano cruza 3 columnas como máximo.
    let gw = 2 + usize::from(hard(rng, difficulty) || rng.chance(0.3));
    flat(level, x0, 2, g, rng);
    for i in 0..gw {
        level.set_ground_column(x0 + 2 + i, HEIGHT);
    }
    flat(level, x0 + 2 + gw, 2, g, rng);
    // Arco de monedas en el pico del salto (Smith et al., 2009).
    let mid = x0 + 2 + gw / 2;
    set(level, mid, g.saturating_sub(4), Tile::Coin);
    if gw >= 3 {
        set(level, mid - 1, g.saturating_sub(3), Tile::Coin);
        set(level, mid + 1, g.saturating_sub(3), Tile::Coin);
    }
    Built {
        width: gw + 4,
        exit: g,
    }
}

/// Dos huecos separados por una isla.
fn build_multi_gap(
    level: &mut Level,
    x0: usize,
    g: usize,
    _difficulty: f64,
    rng: &mut Rng,
) -> Built {
    let (g1, g2) = (2 + rng.index(2), 2 + rng.index(2));
    let island = 2 + rng.index(2);
    flat(level, x0, 2, g, rng);
    let mut x = x0 + 2;
    for _ in 0..g1 {
        level.set_ground_column(x, HEIGHT);
        x += 1;
    }
    let ig = (g as i32 - rng.range_i32(-1, 1)).clamp(MIN_GROUND as i32, MAX_GROUND as i32) as usize;
    for _ in 0..island {
        level.set_ground_column(x, ig);
        x += 1;
    }
    for _ in 0..g2 {
        level.set_ground_column(x, HEIGHT);
        x += 1;
    }
    flat(level, x, 2, g, rng);
    Built {
        width: x + 2 - x0,
        exit: g,
    }
}

/// Escalones de subida o bajada.
fn build_stairs(level: &mut Level, x0: usize, g: usize, difficulty: f64, rng: &mut Rng) -> Built {
    let steps = 2 + rng.index(3);
    // Sube si hay sitio por arriba; baja solo si caben todos los peldaños.
    let room_up = g >= MIN_GROUND + steps;
    let room_down = g + steps <= MAX_GROUND;
    let up = room_up && (!room_down || rng.chance(0.6));
    let steps = if up || room_down {
        steps
    } else {
        (MAX_GROUND - g).max(1)
    };
    let mut gg = g;
    let mut x = x0;
    for _ in 0..steps {
        let rise = 1 + usize::from(hard(rng, difficulty));
        gg = if up {
            gg.saturating_sub(rise).max(MIN_GROUND)
        } else {
            (gg + rise).min(MAX_GROUND)
        };
        let sw = 2 + rng.index(2);
        flat(level, x, sw, gg, rng);
        x += sw;
    }
    Built {
        width: x - x0,
        exit: gg,
    }
}

/// Meseta escalonada con enemigo y monedas.
fn build_mesa(level: &mut Level, x0: usize, g: usize, difficulty: f64, rng: &mut Rng) -> Built {
    let h = 2 + rng.index(2);
    let top = g.saturating_sub(h).max(MIN_GROUND);
    let mid = usize::midpoint(g, top);
    flat(level, x0, 2, mid, rng);
    let plateau = 4 + rng.index(3);
    flat(level, x0 + 2, plateau, top, rng);
    if rng.chance(0.5 + 0.4 * difficulty) {
        set(level, x0 + 2 + plateau / 2 + 1, top - 1, Tile::Enemy);
    }
    coins(level, x0 + 3, 3, top.saturating_sub(2));
    flat(level, x0 + 2 + plateau, 2, mid, rng);
    Built {
        width: plateau + 4,
        exit: g,
    }
}

/// Zanja con un enemigo dentro.
fn build_trench(level: &mut Level, x0: usize, g: usize, difficulty: f64, rng: &mut Rng) -> Built {
    let depth = 2 + usize::from(hard(rng, difficulty));
    let bottom = (g + depth).min(MAX_GROUND);
    let w = 4 + rng.index(2);
    flat(level, x0, 2, g, rng);
    for i in 0..w {
        level.set_ground_column(x0 + 2 + i, bottom);
    }
    set(level, x0 + 2 + w / 2, bottom - 1, Tile::Enemy);
    coins(level, x0 + 3, w - 2, g.saturating_sub(2));
    flat(level, x0 + 2 + w, 2, g, rng);
    Built {
        width: w + 4,
        exit: g,
    }
}

/// Uno o varios enemigos en un tramo llano.
fn build_horde(
    level: &mut Level,
    kind: Kind,
    x0: usize,
    g: usize,
    difficulty: f64,
    rng: &mut Rng,
) -> Built {
    let n = if kind == Kind::Enemy {
        1
    } else {
        2 + usize::from(hard(rng, difficulty))
    };
    // Separados lo justo para aterrizar entre sus zonas de patrulla.
    let step = crate::ga::init::ENEMY_SPACING;
    let w = 4 + step * (n - 1) + 2;
    flat(level, x0, w, g, rng);
    for i in 0..n {
        set(level, x0 + 2 + step * i, g - 1, Tile::Enemy);
    }
    Built { width: w, exit: g }
}

/// Tira de pinchos saltable con monedas encima.
fn build_spikes(level: &mut Level, x0: usize, g: usize, difficulty: f64, rng: &mut Rng) -> Built {
    let n = 1 + rng.index(2) + usize::from(hard(rng, difficulty));
    let w = n + 4;
    flat(level, x0, w, g, rng);
    for i in 0..n {
        set(level, x0 + 2 + i, g - 1, Tile::Hazard);
    }
    coins(level, x0 + 2, n, g.saturating_sub(4));
    Built { width: w, exit: g }
}

/// Hueco largo partido por una plataforma flotante.
fn build_bridge(level: &mut Level, x0: usize, g: usize, difficulty: f64, rng: &mut Rng) -> Built {
    // Hueco largo partido por una plataforma 2 filas por encima del borde:
    // tramos de 2 columnas (Δh = +2 → hueco máximo 2) y bajada libre.
    let pw = 3 + rng.index(2);
    let (a, b) = (2, 2 + usize::from(hard(rng, difficulty)));
    flat(level, x0, 2, g, rng);
    let span = a + pw + b;
    for i in 0..span {
        level.set_ground_column(x0 + 2 + i, HEIGHT);
    }
    let row = g.saturating_sub(2);
    platform(level, x0 + 2 + a, pw, row);
    coins(level, x0 + 2 + a, pw, row.saturating_sub(2));
    flat(level, x0 + 2 + span, 2, g, rng);
    Built {
        width: span + 4,
        exit: g,
    }
}

/// Cadena de plataformas que sube.
fn build_chain(level: &mut Level, x0: usize, g: usize, difficulty: f64, rng: &mut Rng) -> Built {
    // Cadena de 3 a 4 plataformas que sube 2 filas por paso (la primera 3
    // filas sobre el suelo, que deja 2 libres debajo), con huecos de 1-2.
    let n = 3 + rng.index(2);
    let mut row = g.saturating_sub(3);
    let mut x = x0 + 1;
    let mut widths = Vec::with_capacity(n);
    let mut rows = Vec::with_capacity(n);
    for _ in 0..n {
        if row < 2 {
            break;
        }
        let pw = 2 + rng.index(3);
        widths.push((x, pw));
        rows.push(row);
        x += pw + 1 + rng.index(2);
        row = row.saturating_sub(2);
    }
    let w = (x + 1 - x0).min(max_width(Kind::Chain));
    flat(level, x0, w, g, rng);
    for (&(px, pw), &r) in widths.iter().zip(&rows) {
        platform(level, px, pw.min(x0 + w - px), r);
    }
    if let (Some(&(px, pw)), Some(&r)) = (widths.last(), rows.last()) {
        coins(level, px, pw.min(3), r.saturating_sub(1));
    }
    // Variante difícil: pinchos bajo el centro de la cadena.
    if hard(rng, difficulty) {
        for i in 0..2 {
            set(level, x0 + w / 2 + i, g - 1, Tile::Hazard);
        }
    }
    Built { width: w, exit: g }
}

/// Doble ruta de riesgo y recompensa.
fn build_two_path(
    level: &mut Level,
    x0: usize,
    g: usize,
    _difficulty: f64,
    rng: &mut Rng,
) -> Built {
    // Riesgo y recompensa (Dahlskog 2-Path): ruta alta con monedas sobre
    // una ruta baja con enemigo o pinchos.
    let w = 10 + rng.index(3);
    flat(level, x0, w, g, rng);
    let row = g.saturating_sub(3);
    platform(level, x0 + 2, w - 4, row);
    coins(level, x0 + 3, (w - 6).min(5), row.saturating_sub(1));
    if rng.chance(0.5) {
        set(level, x0 + w / 2, g - 1, Tile::Enemy);
    } else {
        for i in 0..2 {
            set(level, x0 + w / 2 + i, g - 1, Tile::Hazard);
        }
    }
    Built { width: w, exit: g }
}

/// Huecos separados por pilares.
fn build_pillar_gap(
    level: &mut Level,
    x0: usize,
    g: usize,
    difficulty: f64,
    rng: &mut Rng,
) -> Built {
    let pillars = 2 + rng.index(2);
    flat(level, x0, 2, g, rng);
    let mut x = x0 + 2;
    for _ in 0..pillars {
        let gw = 2 + usize::from(hard(rng, difficulty));
        for _ in 0..gw {
            level.set_ground_column(x, HEIGHT);
            x += 1;
        }
        let ph = (g as i32 - rng.range_i32(0, 1)).max(MIN_GROUND as i32) as usize;
        level.set_ground_column(x, ph);
        x += 1;
    }
    for _ in 0..2 {
        level.set_ground_column(x, HEIGHT);
        x += 1;
    }
    flat(level, x, 2, g, rng);
    Built {
        width: x + 2 - x0,
        exit: g,
    }
}

/// Desnivel mayor que el salto con escalera de mano.
fn build_ladder_shaft(
    level: &mut Level,
    x0: usize,
    g: usize,
    difficulty: f64,
    rng: &mut Rng,
) -> Built {
    // Desnivel de 4-5 filas (más que el salto) salvado con una escalera.
    let rise = (4 + rng.index(2)).min(g.saturating_sub(MIN_GROUND));
    if rise < 4 {
        return build(level, Kind::Stairs, x0, g, difficulty, rng);
    }
    let top = g - rise;
    flat(level, x0, 3, g, rng);
    for y in top - 1..g {
        level.set(x0 + 2, y, Tile::Ladder);
    }
    flat(level, x0 + 3, 4, top, rng);
    Built {
        width: 7,
        exit: top,
    }
}

/// Suaviza la unión entre la columna `x` (final de un patrón) y la siguiente:
/// si la siguiente está más de 2 filas por encima, eleva las columnas previas
/// en escalones de 2 para que el salto alcance.
pub fn smooth_junction(level: &mut Level, x: usize) {
    if x + 1 >= level.width() {
        return;
    }
    let next = level.ground_top(x + 1);
    if next >= HEIGHT {
        return;
    }
    let mut target = next + 2;
    let mut cx = x;
    while cx >= SAFE_COLS {
        let cur = level.ground_top(cx);
        if cur <= target || cur >= HEIGHT && target >= HEIGHT {
            break;
        }
        let keep: Vec<Tile> = level.column(cx)[..target].to_vec();
        level.set_ground_column(cx, target);
        for (y, t) in keep.into_iter().enumerate() {
            if matches!(t, Tile::Platform | Tile::Coin) {
                level.set(cx, y, t);
            }
        }
        target += 2;
        if cx == 0 {
            break;
        }
        cx -= 1;
    }
}

/// Elige el siguiente patrón según el acto del nivel (estructura en cuatro
/// actos de Nintendo: presentar, desarrollar, giro, conclusión).
pub fn pick(rng: &mut Rng, theme: [Kind; 2], act: usize) -> Kind {
    const ANY: [Kind; 12] = [
        Kind::Gap,
        Kind::MultiGap,
        Kind::Stairs,
        Kind::Mesa,
        Kind::Trench,
        Kind::Horde,
        Kind::Spikes,
        Kind::Bridge,
        Kind::Chain,
        Kind::TwoPath,
        Kind::PillarGap,
        Kind::Enemy,
    ];
    match act {
        // Presentar: la mecánica tema A en su forma fácil. Si el tema usa
        // enemigos, el primero aparece aislado para enseñarlos (Super Mario 1-1).
        0 => {
            if matches!(theme[0], Kind::Horde | Kind::Trench) {
                Kind::Enemy
            } else {
                theme[0]
            }
        }
        // Desarrollar: A o cualquier otro patrón.
        1 => {
            if rng.chance(0.5) {
                theme[0]
            } else {
                ANY[rng.index(ANY.len())]
            }
        }
        // Giro: la mecánica B (o una escalera si hay altura).
        2 => {
            if rng.chance(0.15) {
                Kind::LadderShaft
            } else if rng.chance(0.6) {
                theme[1]
            } else {
                ANY[rng.index(ANY.len())]
            }
        }
        _ => ANY[rng.index(ANY.len())],
    }
}

/// Temas posibles (mecánica A con más presencia y mecánica B para el giro).
pub const THEMES: [[Kind; 2]; 8] = [
    [Kind::Chain, Kind::Horde],
    [Kind::Gap, Kind::Bridge],
    [Kind::TwoPath, Kind::Spikes],
    [Kind::Stairs, Kind::Chain],
    [Kind::Mesa, Kind::Trench],
    [Kind::Bridge, Kind::PillarGap],
    [Kind::Chain, Kind::TwoPath],
    [Kind::MultiGap, Kind::Mesa],
];

// --- Detección ---------------------------------------------------------------

/// Patrón detectado en la matriz.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Found {
    /// Tipo.
    pub kind: Kind,
    /// Columna donde empieza.
    pub x: usize,
}

/// Tramos horizontales de plataforma: `(fila, x0, x1)` inclusive.
fn platform_runs(level: &Level) -> Vec<(usize, usize, usize)> {
    let mut runs = Vec::new();
    for y in 0..HEIGHT {
        let mut x = 0;
        while x < level.width() {
            if level.get(x, y) == Tile::Platform {
                let x0 = x;
                while x < level.width() && level.get(x, y) == Tile::Platform {
                    x += 1;
                }
                runs.push((y, x0, x - 1));
            } else {
                x += 1;
            }
        }
    }
    runs.sort_by_key(|r| (r.1, r.0));
    runs
}

/// Detecta los patrones presentes en un nivel (solo lee la matriz).
#[expect(
    clippy::too_many_lines,
    reason = "canalización lineal de detectores por familia, cada uno comentado"
)]
pub fn detect(level: &Level) -> Vec<Found> {
    let w = level.width();
    let lo = SAFE_COLS;
    let hi = w - SAFE_COLS;
    let tops: Vec<usize> = (0..w).map(|x| level.ground_top(x)).collect();
    let runs = platform_runs(level);
    let over = |x0: usize, x1: usize| runs.iter().any(|&(_, a, b)| a <= x1 && b >= x0);
    let mut out = Vec::new();

    // Huecos, puentes, multihuecos y pilares.
    let mut gaps: Vec<(usize, usize)> = Vec::new();
    let mut x = lo;
    while x < hi {
        if tops[x] == HEIGHT {
            let x0 = x;
            while x < hi && tops[x] == HEIGHT {
                x += 1;
            }
            gaps.push((x0, x - 1));
        } else {
            x += 1;
        }
    }
    let mut i = 0;
    while i < gaps.len() {
        let (a, b) = gaps[i];
        if over(a, b) && b - a + 1 >= 4 {
            out.push(Found {
                kind: Kind::Bridge,
                x: a,
            });
            i += 1;
            continue;
        }
        // Encadena huecos separados por islas cortas.
        let mut j = i;
        let mut narrow_islands = 0;
        while j + 1 < gaps.len() && gaps[j + 1].0 - gaps[j].1 - 1 <= 3 {
            if gaps[j + 1].0 - gaps[j].1 - 1 <= 1 {
                narrow_islands += 1;
            }
            j += 1;
        }
        let kind = if j == i {
            Kind::Gap
        } else if narrow_islands * 2 >= j - i {
            Kind::PillarGap
        } else {
            Kind::MultiGap
        };
        out.push(Found { kind, x: a });
        i = j + 1;
    }

    // Relieve: escalones (≥ 2 cambios del mismo signo cercanos), mesetas y zanjas.
    let mut changes: Vec<(usize, i32)> = Vec::new();
    for x in lo + 1..hi {
        if tops[x] < HEIGHT && tops[x - 1] < HEIGHT && tops[x] != tops[x - 1] {
            changes.push((x, tops[x - 1] as i32 - tops[x] as i32));
        }
    }
    // Grupos de cambios consecutivos del mismo signo: (x inicial, x final, signo, n).
    let mut groups: Vec<(usize, usize, i32, usize)> = Vec::new();
    for &(x, d) in &changes {
        match groups.last_mut() {
            Some(gr) if gr.2 == d.signum() && x - gr.1 <= 4 => {
                gr.1 = x;
                gr.3 += 1;
            }
            _ => groups.push((x, x, d.signum(), 1)),
        }
    }
    let mut k = 0;
    while k < groups.len() {
        let (x0, x1, sign, n) = groups[k];
        // Subida y luego bajada cercanas: meseta; bajada y luego subida: zanja o valle.
        if let Some(&(y0, _, s2, _)) = groups.get(k + 1)
            && s2 == -sign
            && y0 - x1 >= 2
            && y0 - x1 <= 10
        {
            let has_enemy = (x1..y0).any(|cx| {
                tops[cx] > 0 && tops[cx] < HEIGHT && level.get(cx, tops[cx] - 1) == Tile::Enemy
            });
            let kind = if sign > 0 {
                Kind::Mesa
            } else if has_enemy {
                Kind::Trench
            } else {
                Kind::Stairs
            };
            out.push(Found { kind, x: x0 });
            k += 2;
            continue;
        }
        if n >= 2 {
            out.push(Found {
                kind: Kind::Stairs,
                x: x0,
            });
        }
        k += 1;
    }

    // Enemigos (aislados u hordas) y pinchos sobre superficies.
    let mut enemy_x: Vec<usize> = Vec::new();
    let mut spike_x: Vec<usize> = Vec::new();
    for x in lo..hi {
        for y in 1..HEIGHT {
            match level.get(x, y) {
                Tile::Enemy => enemy_x.push(x),
                Tile::Hazard => spike_x.push(x),
                _ => {}
            }
        }
    }
    // Elementos a menos de `reach` columnas forman un mismo grupo.
    let group =
        |xs: &[usize], reach: usize, single: Kind, many: Option<Kind>, out: &mut Vec<Found>| {
            let mut i = 0;
            while i < xs.len() {
                let mut j = i;
                while j + 1 < xs.len() && xs[j + 1] - xs[j] <= reach {
                    j += 1;
                }
                let kind = match many {
                    Some(m) if j > i => m,
                    _ => single,
                };
                out.push(Found { kind, x: xs[i] });
                i = j + 1;
            }
        };
    group(
        &enemy_x,
        crate::ga::init::ENEMY_SPACING + 1,
        Kind::Enemy,
        Some(Kind::Horde),
        &mut out,
    );
    group(&spike_x, 4, Kind::Spikes, None, &mut out);

    // Plataformas: cadenas (≥ 2 tramos encadenados a distinta fila) y dobles
    // rutas (tramo ≥ 4 sobre suelo transitable).
    let mut used = vec![false; runs.len()];
    for a in 0..runs.len() {
        if used[a] {
            continue;
        }
        let mut chain = vec![a];
        let mut last = a;
        for b in a + 1..runs.len() {
            let (ry, _, rb) = runs[last];
            let (sy, sa, _) = runs[b];
            if !used[b] && sa > rb && sa - rb <= 3 && sy != ry && ry.abs_diff(sy) <= 3 {
                chain.push(b);
                last = b;
            }
        }
        if chain.len() >= 2 {
            for &c in &chain {
                used[c] = true;
            }
            out.push(Found {
                kind: Kind::Chain,
                x: runs[a].1,
            });
        }
    }
    for (i, &(y, a, b)) in runs.iter().enumerate() {
        if used[i] || b - a + 1 < 4 {
            continue;
        }
        let ground_below =
            (a..=b).filter(|&x| tops[x] < HEIGHT && tops[x] > y).count() * 2 > b - a + 1;
        if ground_below {
            out.push(Found {
                kind: Kind::TwoPath,
                x: a,
            });
        }
    }

    // Escaleras de mano que salvan más de 3 filas.
    for x in lo..hi {
        let mut y = 0;
        while y < HEIGHT {
            if level.get(x, y) == Tile::Ladder {
                let y0 = y;
                while y < HEIGHT && level.get(x, y) == Tile::Ladder {
                    y += 1;
                }
                if y - y0 >= 4 {
                    out.push(Found {
                        kind: Kind::LadderShaft,
                        x,
                    });
                }
            } else {
                y += 1;
            }
        }
    }

    out.sort_by_key(|f| f.x);
    out
}

/// Zonas de descanso: tramos de ≥ 3 columnas con suelo llano y sin peligros.
pub fn rest_areas(level: &Level) -> Vec<(usize, usize)> {
    let w = level.width();
    let safe = |x: usize| {
        let top = level.ground_top(x);
        top < HEIGHT && top > 0 && !level.get(x, top - 1).is_lethal()
    };
    let mut out = Vec::new();
    let mut x = 0;
    while x < w {
        if safe(x) {
            let x0 = x;
            let h = level.ground_top(x);
            while x < w && safe(x) && level.ground_top(x) == h {
                x += 1;
            }
            if x - x0 >= 3 {
                out.push((x0, x - 1));
            }
        } else {
            x += 1;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::level::Style;
    use crate::play::{Checker, PlayScratch};

    fn base(w: usize) -> Level {
        let mut l = Level::empty(w, Style::Base);
        crate::ga::init::write_safe_zones(&mut l);
        for x in SAFE_COLS..w - SAFE_COLS {
            l.set_ground_column(x, 12);
        }
        l
    }

    #[test]
    fn cada_patron_es_jugable_y_se_detecta() {
        let checker = Checker::default();
        for &kind in &Kind::ALL {
            let mut ok = 0;
            let mut seen = 0;
            for seed in 0..20 {
                let mut rng = Rng::seed_from_u64(seed);
                let mut l = base(40);
                let b = build(&mut l, kind, 8, 12, 0.5, &mut rng);
                assert!(b.width <= max_width(kind), "{kind:?} ocupa {}", b.width);
                // Después del patrón, el suelo continúa a la altura de salida.
                for x in 8 + b.width..40 - SAFE_COLS {
                    l.set_ground_column(x, b.exit);
                }
                smooth_junction(&mut l, 8 + b.width - 1);
                crate::ga::init::write_safe_zones(&mut l);
                smooth_junction(&mut l, 40 - SAFE_COLS - 1);
                if checker.check(&l, &mut PlayScratch::default()).solvable {
                    ok += 1;
                }
                if kind == Kind::Rest
                    || detect(&l)
                        .iter()
                        .any(|f| f.kind == kind || (kind == Kind::Horde && f.kind == Kind::Enemy))
                {
                    seen += 1;
                }
            }
            assert!(ok >= 18, "{kind:?}: solo {ok}/20 jugables");
            assert!(seen >= 14, "{kind:?}: solo detectado {seen}/20");
        }
    }

    #[test]
    fn descansos() {
        let l = base(30);
        let r = rest_areas(&l);
        assert_eq!(r.len(), 1);
        assert!(r[0].1 - r[0].0 >= 20);
    }
}
