//! Métricas de un nivel: rasgos estructurales, dificultad, variedad y coherencia.
//!
//! Convenciones (declaradas explícitamente porque la literatura no es uniforme):
//! - **Leniencia**: mayor = más permisivo (signo invertido respecto a Smith y
//!   Whitehead 2010, como en trabajos posteriores).
//! - **Linealidad**: 1 = superficie perfectamente recta; 0 = muy irregular.
//! - Todas las puntuaciones de objetivos están en `[0, 1]` y se maximizan.

use crate::level::{HEIGHT, Level, SAFE_COLS, Style, Tile};
use crate::play::{PlayResult, PlayScratch};

/// Rasgos estructurales y de juego de un nivel.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Features {
    /// Columnas.
    pub width: usize,
    /// Número de enemigos.
    pub enemies: u32,
    /// Número de peligros.
    pub hazards: u32,
    /// Enemigos y peligros "relevantes": junto a una celda que el jugador
    /// puede atravesar (los inalcanzables no aportan dificultad).
    pub relevant_threats: u32,
    /// Número de monedas.
    pub coins: u32,
    /// Celdas de escalera.
    pub ladders: u32,
    /// Celdas de plataforma.
    pub platforms: u32,
    /// Decoraciones.
    pub decor: u32,
    /// Número de huecos (rachas de columnas sin suelo).
    pub gaps: u32,
    /// Anchura media de los huecos.
    pub gap_mean: f64,
    /// Anchura máxima de hueco.
    pub gap_max: u32,
    /// Saltos necesarios en el camino óptimo.
    pub jumps: u32,
    /// Saltos ajustados.
    pub tight_jumps: u32,
    /// Fracción de celdas vacías alcanzables (espacio negativo).
    pub negative_space: f64,
    /// Linealidad de la superficie del suelo, `[0, 1]`.
    pub linearity: f64,
    /// Leniencia, `[0, 1]` (mayor = más fácil).
    pub leniency: f64,
    /// Dispersión horizontal de enemigos normalizada (Summerville et al. 2017).
    pub enemy_sparsity: f64,
    /// Fracción de celdas sólidas.
    pub density: f64,
    /// Patrones 3×3 distintos / ventanas 3×3.
    pub pattern_ratio: f64,
    /// Entropía normalizada de tipos no vacíos.
    pub entropy: f64,
    /// Variedad entre segmentos, `[0, 1]`.
    pub segment_variety: f64,
    /// Violaciones locales de coherencia.
    pub violations: u32,
    /// Monedas no alcanzables.
    pub unreachable_coins: u32,
    /// Fracción de plataformas y escaleras que el jugador nunca puede tocar.
    pub unreachable_structure: f64,
    /// Celdas de plataforma alcanzables.
    pub reachable_platforms: u32,
    /// Celdas de escalera alcanzables.
    pub reachable_ladders: u32,
}

/// Memoria de trabajo reutilizable de las métricas.
#[derive(Debug, Clone, Default)]
pub struct MetricScratch {
    patterns: Vec<u64>,
    segments: Vec<[f64; 6]>,
}

/// Longitud de segmento para la variedad entre segmentos.
pub const SEGMENT: usize = 16;

fn evolvable_range(level: &Level) -> std::ops::Range<usize> {
    SAFE_COLS..level.width() - SAFE_COLS
}

/// Huecos: rachas de columnas sin suelo dentro de la región evolutiva.
fn gaps(level: &Level) -> (u32, f64, u32) {
    let (mut count, mut total, mut max, mut run) = (0_u32, 0_u32, 0_u32, 0_u32);
    let mut close = |run: &mut u32| {
        if *run > 0 {
            count += 1;
            total += *run;
            max = max.max(*run);
            *run = 0;
        }
    };
    for x in evolvable_range(level) {
        if level.ground_top(x) == HEIGHT {
            run += 1;
        } else {
            close(&mut run);
        }
    }
    close(&mut run);
    let mean = if count == 0 {
        0.0
    } else {
        f64::from(total) / f64::from(count)
    };
    (count, mean, max)
}

/// Linealidad: 1 − desviación absoluta media respecto a la recta de regresión de
/// la altura de la superficie, normalizada por media altura del nivel.
fn linearity(level: &Level) -> f64 {
    let pts: Vec<(f64, f64)> = (0..level.width())
        .filter_map(|x| {
            let top = level.ground_top(x);
            (top < HEIGHT).then_some((x as f64, top as f64))
        })
        .collect();
    if pts.len() < 2 {
        return 1.0;
    }
    let n = pts.len() as f64;
    let (mx, my) = (
        pts.iter().map(|p| p.0).sum::<f64>() / n,
        pts.iter().map(|p| p.1).sum::<f64>() / n,
    );
    let sxx: f64 = pts.iter().map(|p| (p.0 - mx).powi(2)).sum();
    let sxy: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let slope = if sxx > 0.0 { sxy / sxx } else { 0.0 };
    let mad = pts
        .iter()
        .map(|p| (p.1 - (my + slope * (p.0 - mx))).abs())
        .sum::<f64>()
        / n;
    (1.0 - mad / (HEIGHT as f64 / 2.0)).clamp(0.0, 1.0)
}

/// Codifica la ventana 3×3 con esquina en `(x, y)` en 36 bits (4 por tile).
#[inline]
fn pattern(level: &Level, x: usize, y: usize) -> u64 {
    let mut key = 0_u64;
    for dx in 0..3 {
        for &t in &level.column(x + dx)[y..y + 3] {
            key = (key << 4) | u64::from(t as u8);
        }
    }
    key
}

/// Patrones 3×3 distintos / ventanas (sin `HashSet`: ordenar + deduplicar es
/// determinista y no depende de la semilla aleatoria de `RandomState`).
fn pattern_ratio(level: &Level, buf: &mut Vec<u64>) -> f64 {
    buf.clear();
    for x in 0..level.width() - 2 {
        for y in 0..HEIGHT - 2 {
            buf.push(pattern(level, x, y));
        }
    }
    let windows = buf.len();
    buf.sort_unstable();
    buf.dedup();
    buf.len() as f64 / windows.max(1) as f64
}

/// Entropía de Shannon de los tipos no vacíos evolucionables, normalizada a `[0, 1]`.
fn entropy(counts: &[u32; Tile::COUNT]) -> f64 {
    let kinds = [
        Tile::Ground,
        Tile::Platform,
        Tile::Ladder,
        Tile::Hazard,
        Tile::Enemy,
        Tile::Coin,
        Tile::Decor,
    ];
    let total: u32 = kinds.iter().map(|&t| counts[t as usize]).sum();
    if total == 0 {
        return 0.0;
    }
    let h: f64 = kinds
        .iter()
        .map(|&t| f64::from(counts[t as usize]) / f64::from(total))
        .filter(|&p| p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    h / (kinds.len() as f64).ln()
}

/// Variedad entre segmentos: distancia L1 media entre vectores de rasgos de
/// segmentos de [`SEGMENT`] columnas, con penalización por segmentos
/// consecutivos idénticos.
fn segment_variety(level: &Level, segs: &mut Vec<[f64; 6]>) -> f64 {
    segs.clear();
    let range = evolvable_range(level);
    let mut x = range.start;
    while x + SEGMENT <= range.end {
        let mut f = [0.0; 6];
        for cx in x..x + SEGMENT {
            let top = level.ground_top(cx);
            f[0] += top as f64 / HEIGHT as f64;
            f[1] += f64::from(u8::from(top == HEIGHT));
            for &t in level.column(cx) {
                match t {
                    Tile::Enemy | Tile::Hazard => f[2] += 1.0,
                    Tile::Platform => f[3] += 1.0,
                    Tile::Coin => f[4] += 1.0,
                    Tile::Ladder => f[5] += 1.0,
                    _ => {}
                }
            }
        }
        // Normaliza: alturas y huecos por columna; objetos con techo razonable.
        let k = SEGMENT as f64;
        let scale = [k, k, k * 0.3, k * 1.0, k * 0.5, k * 0.5];
        for (v, s) in f.iter_mut().zip(scale) {
            *v = (*v / s).min(1.0);
        }
        segs.push(f);
        x += SEGMENT;
    }
    if segs.len() < 2 {
        return 0.0;
    }
    let dist =
        |a: &[f64; 6], b: &[f64; 6]| a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum::<f64>() / 6.0;
    let mut sum = 0.0;
    let mut pairs = 0.0;
    for i in 0..segs.len() {
        for j in i + 1..segs.len() {
            sum += dist(&segs[i], &segs[j]);
            pairs += 1.0;
        }
    }
    let identical = segs
        .windows(2)
        .filter(|w| dist(&w[0], &w[1]) < 1e-9)
        .count() as f64;
    // Una distancia media de 0,25 ya es mucha variedad: se escala a [0, 1].
    ((sum / pairs) / 0.25 - 0.1 * identical).clamp(0.0, 1.0)
}

/// Defectos de un tramo de escalera, evaluados en su celda superior: tramo de
/// un solo tile o escalera sin salida (arriba no se puede estar de pie ni a los
/// lados ni encima). Cuenta una vez por tramo.
fn ladder_flaw(level: &Level, x: i32, y: i32) -> bool {
    if level.get_or_empty(x, y - 1) == Tile::Ladder {
        return false; // no es la celda superior del tramo
    }
    let short = level.get_or_empty(x, y + 1) != Tile::Ladder;
    let stand = |cx: i32, cy: i32| {
        cy >= 0
            && level.get_or_empty(cx, cy).is_passable()
            && level.get_or_empty(cx, cy + 1).is_support()
    };
    // Desde lo alto se sale a un lado, o hacia arriba si encima hay hueco y apoyo cerca.
    let exit = stand(x - 1, y) || stand(x + 1, y) || stand(x - 1, y - 1) || stand(x + 1, y - 1);
    short || !exit
}

/// Cuenta violaciones locales de coherencia visual/lógica.
fn violations(level: &Level) -> u32 {
    let mut v = 0;
    let w = level.width() as i32;
    for x in 0..w {
        for y in 0..HEIGHT as i32 {
            let t = level.get_or_empty(x, y);
            let below = level.get_or_empty(x, y + 1);
            let solid_below = below.is_solid() || y + 1 == HEIGHT as i32;
            v += u32::from(match t {
                // Pinchos, enemigos y decoración deben apoyarse en algo sólido.
                Tile::Hazard | Tile::Enemy | Tile::Decor => !solid_below,
                // El suelo es terreno: debe llegar hasta el fondo (sin "islas de tierra").
                Tile::Ground => !(below == Tile::Ground || y + 1 == HEIGHT as i32),
                // Plataforma de un solo tile sin vecinas (ruido visual) o
                // bloque flotante grueso (plataforma sobre plataforma).
                Tile::Platform => {
                    (level.get_or_empty(x - 1, y) != Tile::Platform
                        && level.get_or_empty(x + 1, y) != Tile::Platform)
                        || below == Tile::Platform
                }
                // La escalera debe tocar apoyo por abajo (otra escalera o sólido).
                Tile::Ladder => {
                    !(below.is_solid() || below == Tile::Ladder) || ladder_flaw(level, x, y)
                }
                // Moneda "perdida en el cielo": sin apoyo a 3 celdas o menos debajo.
                Tile::Coin => !(1..=3).any(|d| level.get_or_empty(x, y + d).is_support()),
                _ => false,
            });
            // Nada en la fila superior: queda cortado por el borde de la pantalla.
            v += u32::from(y == 0 && t != Tile::Empty);
        }
    }
    v
}

/// Fracción de celdas de plataforma o escalera que el jugador no puede tocar:
/// una plataforma cuenta como alcanzable si alguna celda vecina (3×3) es
/// atravesable por el jugador; una escalera, si se atraviesa ella misma.
fn unreachable_structure(level: &Level, reach: &PlayScratch) -> (f64, u32, u32) {
    let (mut total, mut dead, mut plat, mut lad) = (0_u32, 0_u32, 0_u32, 0_u32);
    for x in 0..level.width() as i32 {
        for y in 0..HEIGHT as i32 {
            let reachable = match level.get_or_empty(x, y) {
                Tile::Platform => (-1..=1).any(|dx| {
                    (-1..=1).any(|dy| (dx, dy) != (0, 0) && reach.touched(x + dx, y + dy))
                }),
                Tile::Ladder => reach.touched(x, y),
                _ => continue,
            };
            total += 1;
            dead += u32::from(!reachable);
            if reachable {
                if level.get_or_empty(x, y) == Tile::Platform {
                    plat += 1;
                } else {
                    lad += 1;
                }
            }
        }
    }
    let frac = if total == 0 {
        0.0
    } else {
        f64::from(dead) / f64::from(total)
    };
    (frac, plat, lad)
}

impl Features {
    /// Calcula los rasgos a partir del nivel y del resultado del verificador.
    pub fn compute(
        level: &Level,
        play: &PlayResult,
        reach: &PlayScratch,
        scratch: &mut MetricScratch,
    ) -> Self {
        let mut relevant_threats = 0;
        for x in 0..level.width() as i32 {
            for y in 0..HEIGHT as i32 {
                if level.get_or_empty(x, y).is_lethal()
                    && [(0, -1), (-1, 0), (1, 0), (0, 1), (0, -2)]
                        .iter()
                        .any(|&(dx, dy)| reach.touched(x + dx, y + dy))
                {
                    relevant_threats += 1;
                }
            }
        }
        let mut counts = [0_u32; Tile::COUNT];
        for &t in &level.tiles()[SAFE_COLS * HEIGHT..(level.width() - SAFE_COLS) * HEIGHT] {
            counts[t as usize] += 1;
        }
        let (gaps, gap_mean, gap_max) = gaps(level);
        let width = level.width();
        let cells = (width * HEIGHT) as f64;
        let solid = level.tiles().iter().filter(|t| t.is_solid()).count() as f64;
        let empty = level
            .tiles()
            .iter()
            .filter(|t| !t.is_solid() && !t.is_lethal())
            .count() as f64;
        let enemies = counts[Tile::Enemy as usize];
        let hazards = counts[Tile::Hazard as usize];
        // Dispersión de enemigos: desviación absoluta media de su x, / (ancho/4).
        let ex: Vec<f64> = (0..width)
            .flat_map(|x| {
                level
                    .column(x)
                    .iter()
                    .filter(|&&t| t == Tile::Enemy)
                    .map(move |_| x as f64)
            })
            .collect();
        let enemy_sparsity = if ex.is_empty() {
            0.0
        } else {
            let m = ex.iter().sum::<f64>() / ex.len() as f64;
            (ex.iter().map(|x| (x - m).abs()).sum::<f64>() / ex.len() as f64 / (width as f64 / 4.0))
                .min(1.0)
        };
        let harm = f64::from(gaps + enemies + hazards);
        let structure = unreachable_structure(level, reach);
        Self {
            width,
            enemies,
            hazards,
            relevant_threats,
            coins: counts[Tile::Coin as usize],
            ladders: counts[Tile::Ladder as usize],
            platforms: counts[Tile::Platform as usize],
            decor: counts[Tile::Decor as usize],
            gaps,
            gap_mean,
            gap_max,
            jumps: play.jumps,
            tight_jumps: play.tight_jumps,
            negative_space: if empty > 0.0 {
                f64::from(play.reachable_cells) / empty
            } else {
                0.0
            },
            linearity: linearity(level),
            leniency: (1.0 - harm / (width as f64 / 4.0)).clamp(0.0, 1.0),
            enemy_sparsity,
            density: solid / cells,
            pattern_ratio: pattern_ratio(level, &mut scratch.patterns),
            entropy: entropy(&counts),
            segment_variety: segment_variety(level, &mut scratch.segments),
            violations: violations(level),
            unreachable_coins: counts[Tile::Coin as usize].saturating_sub(play.reachable_coins),
            unreachable_structure: structure.0,
            reachable_platforms: structure.1,
            reachable_ladders: structure.2,
        }
    }

    /// Valor de un rasgo por columna.
    fn per_col(&self, v: u32) -> f64 {
        f64::from(v) / self.width as f64
    }
}

/// Banda suave: 1 dentro de `[lo, hi]` y decae gaussianamente fuera con escala `w`.
pub fn band(v: f64, lo: f64, hi: f64, w: f64) -> f64 {
    let d = if v < lo {
        lo - v
    } else if v > hi {
        v - hi
    } else {
        0.0
    };
    (-(d / w).powi(2) / 2.0).exp()
}

/// Campana gaussiana centrada en `target` con desviación `sigma`.
pub fn bell(v: f64, target: f64, sigma: f64) -> f64 {
    (-((v - target) / sigma).powi(2) / 2.0).exp()
}

/// Parámetros de las métricas objetivo.
#[derive(Debug, Clone, PartialEq)]
pub struct Targets {
    /// Dificultad objetivo `D*` ("moderada").
    pub difficulty: f64,
    /// Tolerancia gaussiana de la dificultad.
    pub difficulty_sigma: f64,
    /// Proporción objetivo de patrones 3×3 distintos `R*`.
    pub pattern_ratio: f64,
    /// Entropía objetivo de tipos.
    pub entropy: f64,
}

impl Default for Targets {
    fn default() -> Self {
        Self {
            difficulty: 0.5,
            difficulty_sigma: 0.15,
            // Calibrados con los 15 niveles de Super Mario Bros del VGLC
            // (`nivelga corpus`): patrones 3×3 distintos con mediana 0,056 en
            // niveles de unas 187 columnas (algo más en niveles más cortos como
            // los nuestros) y entropía de tipos con mediana 0,39.
            pattern_ratio: 0.08,
            entropy: 0.42,
        }
    }
}

/// Dificultad compuesta `D ∈ [0, 1]`. El peso mayor es para los enemigos, el
/// mejor predictor de dificultad percibida (ρ = 0,72; Summerville et al. 2017).
pub fn difficulty(f: &Features, max_jump: f64) -> f64 {
    // Enemigos y peligros cuentan en proporción a los que son relevantes.
    let total = f.enemies + f.hazards;
    let relevance = if total == 0 {
        1.0
    } else {
        f64::from(f.relevant_threats) / f64::from(total)
    };
    let n_e = (f.per_col(f.enemies) * relevance / 0.15).min(1.0);
    let n_h = (f.per_col(f.hazards) * relevance / 0.10).min(1.0);
    let n_g = (f.per_col(f.gaps) / 0.08).min(1.0);
    let w_g = (f.gap_mean / max_jump).min(1.0);
    let n_j = (f64::from(f.jumps) / (f.width as f64 / 5.0)).min(1.0);
    let tau = if f.jumps == 0 {
        0.0
    } else {
        f64::from(f.tight_jumps) / f64::from(f.jumps)
    };
    0.35 * n_e + 0.15 * n_h + 0.15 * n_g + 0.10 * w_g + 0.15 * n_j + 0.10 * tau
}

/// Variedad `V ∈ [0, 1]`.
pub fn variety(f: &Features, t: &Targets) -> f64 {
    let f_pat = bell(f.pattern_ratio, t.pattern_ratio, 0.04);
    let f_ent = bell(f.entropy, t.entropy, 0.12);
    0.4 * f_pat + 0.3 * f_ent + 0.3 * f.segment_variety
}

/// Encaje de la estructura con la "personalidad" del pack de Kenney elegido,
/// `[0, 1]`. Cada estilo declara rangos preferidos de densidad por columna.
pub fn style_fit(f: &Features, style: Style) -> f64 {
    // Solo cuenta la estructura alcanzable: una plataforma a la que no se puede
    // llegar no aporta a la personalidad del estilo (antes la premiaba).
    let plat = f.per_col(f.reachable_platforms);
    let lad = f.per_col(f.reachable_ladders);
    let haz = f.per_col(f.hazards);
    let coin = f.per_col(f.coins);
    let dec = f.per_col(f.decor);
    // (valor, mínimo, máximo, escala de caída). Rangos por columna anclados en
    // el corpus VGLC de Super Mario Bros (`nivelga corpus`): plataformas con
    // mediana 0,38 [0,23; 0,54], monedas 0,05 [0; 0,13] y enemigos 0,07. Las
    // escaleras no existen en ese corpus: se mantienen escasas en todos los
    // estilos (la versión anterior pedía hasta 0,4 por columna en industrial y
    // llenaba los niveles de escaleras sin sentido).
    let terms: [(f64, f64, f64, f64); 5] = match style {
        // Pradera clásica: cercana al corpus, algo de todo.
        Style::Base => [
            (plat, 0.2, 0.55, 0.15),
            (lad, 0.0, 0.02, 0.03),
            (haz, 0.01, 0.05, 0.03),
            (coin, 0.03, 0.12, 0.05),
            (dec, 0.05, 0.15, 0.06),
        ],
        // Granja: terreno abierto, mucha vegetación, pocos peligros.
        Style::Farm => [
            (plat, 0.05, 0.25, 0.12),
            (lad, 0.0, 0.02, 0.03),
            (haz, 0.0, 0.02, 0.03),
            (coin, 0.03, 0.10, 0.05),
            (dec, 0.15, 0.30, 0.08),
        ],
        // Dulce: islas flotantes y más coleccionables.
        Style::Food => [
            (plat, 0.5, 1.0, 0.2),
            (lad, 0.0, 0.02, 0.03),
            (haz, 0.02, 0.05, 0.03),
            (coin, 0.10, 0.20, 0.06),
            (dec, 0.05, 0.12, 0.06),
        ],
        // Industrial: más vertical, con vigas, alguna escalera y más peligros.
        Style::Industrial => [
            (plat, 0.45, 0.9, 0.2),
            (lad, 0.03, 0.08, 0.04),
            (haz, 0.04, 0.08, 0.03),
            (coin, 0.02, 0.08, 0.04),
            (dec, 0.03, 0.10, 0.06),
        ],
    };
    terms
        .iter()
        .map(|&(v, lo, hi, w)| band(v, lo, hi, w))
        .sum::<f64>()
        / terms.len() as f64
}

/// Coherencia `C ∈ [0, 1]`: reglas locales, estructura alcanzable, monedas
/// alcanzables y encaje de estilo.
pub fn coherence(f: &Features, style: Style) -> f64 {
    // Penalización que nunca se satura y que castiga desde la primera
    // violación. Historia: con un tope lineal (0 desde W/4 violaciones) el ruido
    // formaba una meseta; con una forma cuadrática suave un solo defecto (una
    // escalera que no lleva a nada) apenas costaba un 0,5 %.
    let k = f.width as f64 / 30.0;
    let f_rules = 1.0 / (1.0 + f64::from(f.violations) / k);
    let f_coins = if f.coins == 0 {
        1.0
    } else {
        1.0 - f64::from(f.unreachable_coins) / f64::from(f.coins)
    };
    let f_reach = 1.0 - f.unreachable_structure;
    0.45 * f_rules + 0.25 * f_reach + 0.10 * f_coins + 0.20 * style_fit(f, style)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::play::Checker;

    fn flat(width: usize) -> Level {
        let mut l = Level::empty(width, Style::Base);
        for x in 0..width {
            l.set(x, HEIGHT - 1, Tile::Ground);
        }
        l.set(1, HEIGHT - 2, Tile::Start);
        l.set(width - 2, HEIGHT - 2, Tile::Goal);
        l
    }

    fn feats(l: &Level) -> Features {
        let mut sc = PlayScratch::default();
        let p = Checker::default().check(l, &mut sc);
        Features::compute(l, &p, &sc, &mut MetricScratch::default())
    }

    #[test]
    fn nivel_plano_es_facil_lineal_y_poco_variado() {
        let f = feats(&flat(64));
        assert_eq!(f.gaps, 0);
        assert!((f.linearity - 1.0).abs() < 1e-12);
        assert!(difficulty(&f, 4.0) < 0.05);
        assert!(f.segment_variety < 0.05);
        assert_eq!(f.violations, 0);
    }

    #[test]
    fn huecos_y_violaciones() {
        let mut l = flat(64);
        for x in 10..13 {
            l.set(x, HEIGHT - 1, Tile::Empty);
        }
        l.set(20, 3, Tile::Hazard); // pincho flotante
        l.set(30, 3, Tile::Ground); // isla de tierra
        let f = feats(&l);
        assert_eq!((f.gaps, f.gap_max), (1, 3));
        assert_eq!(f.violations, 2);
    }

    #[test]
    fn el_ruido_puntua_peor_que_un_nivel_ordenado() {
        let mut rng = crate::rng::Rng::seed_from_u64(1);
        let ordenado = flat(120);
        let mut ruido = crate::ga::init::uniform_level(120, &mut rng);
        crate::ga::init::repair(&mut ruido, false);
        let (fo, fr) = (feats(&ordenado), feats(&ruido));
        assert!(coherence(&fo, Style::Base) > coherence(&fr, Style::Base) + 0.3);
        // Más violaciones siempre restan (sin meseta).
        let mut a = fo.clone();
        a.violations = 100;
        let mut b = fo.clone();
        b.violations = 200;
        assert!(coherence(&a, Style::Base) > coherence(&b, Style::Base));
    }

    #[test]
    fn escaleras_defectuosas() {
        let mut l = flat(30);
        // Escalera de un solo tile junto al suelo: defecto (tramo corto).
        l.set(5, HEIGHT - 2, Tile::Ladder);
        assert_eq!(feats(&l).violations, 1);
        // Escalera de 4 que sube a una plataforma: correcta.
        let mut l = flat(30);
        for y in HEIGHT - 5..HEIGHT - 1 {
            l.set(10, y, Tile::Ladder);
        }
        l.set(11, HEIGHT - 4, Tile::Platform);
        l.set(12, HEIGHT - 4, Tile::Platform);
        assert_eq!(feats(&l).violations, 0);
        // La misma escalera sin nada arriba: no lleva a ningún sitio.
        l.set(11, HEIGHT - 4, Tile::Empty);
        l.set(12, HEIGHT - 4, Tile::Empty);
        assert_eq!(feats(&l).violations, 1);
    }

    #[test]
    fn estructura_inalcanzable() {
        let mut l = flat(30);
        for x in 10..13 {
            l.set(x, 1, Tile::Platform); // muy alta: inalcanzable
        }
        assert!(feats(&l).unreachable_structure > 0.99);
        let mut l = flat(30);
        for x in 10..13 {
            l.set(x, HEIGHT - 4, Tile::Platform); // a 2 de altura: alcanzable
        }
        assert!(feats(&l).unreachable_structure < 1e-9);
    }

    #[test]
    fn bandas() {
        assert!((band(0.5, 0.4, 0.6, 0.1) - 1.0).abs() < 1e-12);
        assert!(band(0.8, 0.4, 0.6, 0.1) < 0.2);
        assert!((bell(0.5, 0.5, 0.1) - 1.0).abs() < 1e-12);
    }
}
