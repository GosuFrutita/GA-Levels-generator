//! Dibujo de niveles con los tilesets *Pixel Platformer* de Kenney (CC0).
//!
//! Cada [`Style`] define un "skin": qué tile del atlas corresponde a cada tipo
//! lógico, con *autotiling* de bordes (tile izquierdo/central/derecho/aislado
//! según los vecinos) para suelo y plataformas.

use crate::error::Result;
use crate::image::{Image, Rgba, png};
use crate::level::{HEIGHT, Level, Style, Tile};
use std::path::Path;

/// Tamaño de los tiles del terreno, en píxeles.
pub const TILE: u32 = 18;
/// Tamaño de los tiles de personajes y fondos, en píxeles.
pub const BIG: u32 = 24;

/// Hoja de sprites de origen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Sheet {
    Base,
    Farm,
    Food,
    Industrial,
    Characters,
    Backgrounds,
}

impl Sheet {
    /// Archivo de la hoja relativo al directorio de recursos.
    const fn file(self) -> &'static str {
        match self {
            Self::Base => "kenney_pixel-platformer/tilemap_packed.png",
            Self::Farm => "kenney_pixel-platformer-farm-expansion/tilemap_packed.png",
            Self::Food => "kenney_pixel-platformer-food-expansion/tilemap_packed.png",
            Self::Industrial => "kenney_pixel-platformer-industrial-expansion/tilemap_packed.png",
            Self::Characters => "kenney_pixel-platformer/tilemap-characters_packed.png",
            Self::Backgrounds => "kenney_pixel-platformer/tilemap-backgrounds_packed.png",
        }
    }

    /// Columnas de la hoja.
    const fn columns(self) -> u32 {
        match self {
            Self::Base => 20,
            Self::Farm | Self::Food | Self::Industrial => 16,
            Self::Characters => 9,
            Self::Backgrounds => 8,
        }
    }

    /// Lado del sprite en píxeles.
    const fn size(self) -> u32 {
        match self {
            Self::Characters | Self::Backgrounds => BIG,
            _ => TILE,
        }
    }
}

/// Sprite exportado para otros programas (p. ej. el juego en Godot).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpriteRef {
    /// Hoja relativa al directorio de recursos.
    pub file: &'static str,
    /// Índice dentro de la hoja.
    pub index: u16,
    /// Columnas de la hoja.
    pub columns: u32,
    /// Lado del sprite en píxeles.
    pub size: u32,
}

impl From<Spr> for SpriteRef {
    fn from(s: Spr) -> Self {
        Self {
            file: s.0.file(),
            index: s.1,
            columns: s.0.columns(),
            size: s.0.size(),
        }
    }
}

/// Sprites de los elementos dinámicos de un estilo: monedas, enemigo (dos
/// fotogramas), jugador (dos fotogramas) y bandera de meta (asta y dos
/// fotogramas de la tela).
pub fn dynamic_sprites(style: Style) -> Vec<(&'static str, Vec<SpriteRef>)> {
    let sk = skin(style);
    // Segundo fotograma de la animación (la bola de pinchos industrial solo tiene uno).
    let next = |s: Spr| {
        if style == Style::Industrial {
            s
        } else {
            Spr(s.0, s.1 + 1)
        }
    };
    vec![
        ("monedas", sk.coin.iter().map(|&s| s.into()).collect()),
        ("enemigo", vec![sk.enemy.into(), next(sk.enemy).into()]),
        (
            "jugador",
            vec![
                Spr(Sheet::Characters, 0).into(),
                Spr(Sheet::Characters, 1).into(),
            ],
        ),
        (
            "meta",
            vec![
                Spr(Sheet::Base, 131).into(),
                Spr(Sheet::Base, 111).into(),
                Spr(Sheet::Base, 112).into(),
            ],
        ),
    ]
}

/// Escenario estático de un nivel (sin enemigos, monedas, salida ni meta, que
/// se dibujan aparte porque se mueven o desaparecen).
pub fn render_static(level: &Level, sets: &Tilesets) -> Image {
    let mut stat = level.clone();
    for t in stat.tiles_mut() {
        if matches!(*t, Tile::Enemy | Tile::Coin | Tile::Start | Tile::Goal) {
            *t = Tile::Empty;
        }
    }
    render(
        &stat,
        sets,
        &RenderOptions {
            scale: 1,
            path: vec![],
        },
    )
}

/// Referencia a un sprite: hoja + índice.
#[derive(Debug, Clone, Copy)]
struct Spr(Sheet, u16);

/// Cuatro variantes según vecinos horizontales: aislado, izquierda, centro, derecha.
#[derive(Debug, Clone, Copy)]
struct Edge4([Spr; 4]);

impl Edge4 {
    fn pick(self, left: bool, right: bool) -> Spr {
        match (left, right) {
            (false, false) => self.0[0],
            (false, true) => self.0[1],
            (true, true) => self.0[2],
            (true, false) => self.0[3],
        }
    }
}

/// Correspondencia entre tipos lógicos y sprites para un estilo.
#[derive(Debug, Clone)]
struct Skin {
    /// Suelo de una sola fila de alto (borde arriba y abajo).
    ground_single: Edge4,
    /// Superficie que continúa hacia abajo.
    ground_top: Edge4,
    /// Relleno interior.
    ground_fill: Edge4,
    /// Última fila con borde inferior (el suelo termina sobre aire).
    ground_bottom: Edge4,
    platform: Edge4,
    ladder_top: Spr,
    ladder: Spr,
    hazard: &'static [Spr],
    coin: &'static [Spr],
    decor: &'static [Spr],
    enemy: Spr,
    /// Fondos: cielo, horizonte, base (índices en la hoja de fondos).
    background: [u16; 3],
}

const fn e4(sheet: Sheet, a: u16, b: u16, c: u16, d: u16) -> Edge4 {
    Edge4([Spr(sheet, a), Spr(sheet, b), Spr(sheet, c), Spr(sheet, d)])
}

fn skin(style: Style) -> Skin {
    use Sheet::{Base, Characters, Farm, Food, Industrial};
    match style {
        Style::Base => Skin {
            ground_single: e4(Base, 0, 1, 2, 3),
            ground_top: e4(Base, 20, 21, 22, 23),
            ground_fill: e4(Base, 120, 121, 122, 123),
            ground_bottom: e4(Base, 140, 141, 142, 143),
            platform: e4(Base, 48, 48, 49, 50),
            ladder_top: Spr(Base, 51),
            ladder: Spr(Base, 71),
            hazard: &[Spr(Base, 68)],
            coin: &[Spr(Base, 151)],
            decor: &[
                Spr(Base, 124),
                Spr(Base, 125),
                Spr(Base, 126),
                Spr(Base, 128),
            ],
            enemy: Spr(Characters, 15),
            background: [0, 9, 16],
        },
        Style::Farm => Skin {
            ground_single: e4(Farm, 0, 1, 2, 3),
            ground_top: e4(Farm, 16, 17, 18, 19),
            ground_fill: e4(Farm, 32, 33, 34, 35),
            ground_bottom: e4(Farm, 48, 49, 50, 51),
            platform: e4(Farm, 6, 6, 7, 8),
            ladder_top: Spr(Base, 51),
            ladder: Spr(Base, 71),
            hazard: &[Spr(Base, 68)],
            coin: &[Spr(Farm, 4), Spr(Farm, 72)],
            decor: &[Spr(Farm, 20), Spr(Farm, 57), Spr(Farm, 58), Spr(Farm, 43)],
            enemy: Spr(Characters, 24),
            background: [6, 14, 22],
        },
        Style::Food => Skin {
            ground_single: e4(Food, 4, 5, 6, 7),
            ground_top: e4(Food, 20, 21, 22, 23),
            ground_fill: e4(Food, 36, 37, 38, 39),
            ground_bottom: e4(Food, 52, 53, 54, 55),
            platform: e4(Food, 10, 10, 11, 12),
            ladder_top: Spr(Base, 51),
            ladder: Spr(Base, 71),
            hazard: &[Spr(Food, 80), Spr(Food, 81)],
            coin: &[Spr(Food, 13), Spr(Food, 14), Spr(Food, 15)],
            decor: &[Spr(Food, 42), Spr(Food, 43), Spr(Food, 106), Spr(Food, 107)],
            enemy: Spr(Characters, 13),
            background: [2, 11, 18],
        },
        Style::Industrial => Skin {
            ground_single: e4(Industrial, 0, 1, 2, 3),
            ground_top: e4(Industrial, 16, 17, 18, 19),
            ground_fill: e4(Industrial, 32, 33, 34, 35),
            ground_bottom: e4(Industrial, 48, 49, 50, 51),
            platform: e4(Industrial, 4, 4, 5, 6),
            ladder_top: Spr(Industrial, 11),
            ladder: Spr(Industrial, 27),
            // El ácido (13) es un tile de estanque y se leía como un bloque sólido.
            hazard: &[Spr(Base, 68)],
            coin: &[Spr(Base, 151)],
            // La caja rayada (62) parecía sólida: se usan luces de alarma.
            decor: &[
                Spr(Industrial, 9),
                Spr(Industrial, 42),
                Spr(Industrial, 64),
                Spr(Industrial, 65),
            ],
            enemy: Spr(Characters, 8),
            background: [4, 12, 20],
        },
    }
}

/// Atlas cargados en memoria, recortados por tile.
#[derive(Debug, Clone)]
pub struct Tilesets {
    sheets: Vec<Vec<Image>>,
}

fn slice_sheet(img: &Image, size: u32) -> Vec<Image> {
    let (cols, rows) = (img.width() / size, img.height() / size);
    let mut out = Vec::with_capacity((cols * rows) as usize);
    for r in 0..rows {
        for c in 0..cols {
            out.push(img.crop(c * size, r * size, size, size));
        }
    }
    out
}

impl Tilesets {
    /// Carga los atlas `*_packed.png` desde el directorio de recursos.
    ///
    /// # Errors
    ///
    /// Si falta algún archivo o no es un PNG válido.
    pub fn load(assets: impl AsRef<Path>) -> Result<Self> {
        let a = assets.as_ref();
        let base = a.join("kenney_pixel-platformer");
        let exp = |name: &str| {
            a.join(format!("kenney_pixel-platformer-{name}-expansion"))
                .join("tilemap_packed.png")
        };
        let sheets = vec![
            slice_sheet(&png::load(base.join("tilemap_packed.png"))?, TILE),
            slice_sheet(&png::load(exp("farm"))?, TILE),
            slice_sheet(&png::load(exp("food"))?, TILE),
            slice_sheet(&png::load(exp("industrial"))?, TILE),
            slice_sheet(&png::load(base.join("tilemap-characters_packed.png"))?, BIG),
            slice_sheet(
                &png::load(base.join("tilemap-backgrounds_packed.png"))?,
                BIG,
            ),
        ];
        Ok(Self { sheets })
    }

    fn sprite(&self, s: Spr) -> &Image {
        &self.sheets[s.0 as usize][usize::from(s.1)]
    }
}

/// Opciones de dibujo.
#[derive(Debug, Clone)]
pub struct RenderOptions {
    /// Factor de escala entero.
    pub scale: u32,
    /// Trayectoria a superponer, como celdas `(x, y)` en orden
    /// (ver [`crate::play::Checker::trajectory`]).
    pub path: Vec<(usize, usize)>,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            scale: 2,
            path: Vec::new(),
        }
    }
}

/// Elige una variante de una lista de forma determinista según la posición.
fn variant(list: &[Spr], x: usize, y: usize) -> Spr {
    let h = (x.wrapping_mul(73_856_093) ^ y.wrapping_mul(19_349_663)) % list.len();
    list[h]
}

/// Dibuja un nivel.
pub fn render(level: &Level, sets: &Tilesets, opts: &RenderOptions) -> Image {
    let sk = skin(level.style);
    let s = opts.scale.max(1);
    let t = i64::from(TILE * s);
    let (w, h) = (level.width() as u32 * TILE * s, HEIGHT as u32 * TILE * s);
    let mut img = Image::new(w, h, [0, 0, 0, 255]);

    // Fondo: los tiles de fondo son de 24 px; se escalan para cubrir el nivel en bandas.
    let bg_rows = h.div_ceil(BIG * s);
    for by in 0..bg_rows {
        // Cielo arriba, horizonte en la penúltima banda y base en la última.
        let idx = match (by + 2).cmp(&bg_rows) {
            std::cmp::Ordering::Less => sk.background[0],
            std::cmp::Ordering::Equal => sk.background[1],
            std::cmp::Ordering::Greater => sk.background[2],
        };
        for bx in 0..w.div_ceil(BIG * s) {
            let spr = sets.sprite(Spr(Sheet::Backgrounds, idx));
            img.draw(spr, i64::from(bx * BIG * s), i64::from(by * BIG * s), s);
        }
    }

    let is = |x: i32, y: i32, f: fn(Tile) -> bool| f(level.get_or_empty(x, y));
    for x in 0..level.width() {
        for y in 0..HEIGHT {
            let (xi, yi) = (x as i32, y as i32);
            let (px, py) = (x as i64 * t, y as i64 * t);
            let tile = level.get(x, y);
            let spr = match tile {
                Tile::Empty => None,
                Tile::Ground => {
                    // Fuera del mapa por los lados o por abajo, el suelo continúa.
                    let ground = |gx: i32, gy: i32| {
                        gx < 0
                            || gx >= level.width() as i32
                            || gy >= HEIGHT as i32
                            || level.get_or_empty(gx, gy) == Tile::Ground
                    };
                    let (l, r) = (ground(xi - 1, yi), ground(xi + 1, yi));
                    let open_top = yi == 0 || !level.get(x, y - 1).is_solid();
                    let open_bottom = !ground(xi, yi + 1);
                    let set = match (open_top, open_bottom) {
                        (true, true) => sk.ground_single,
                        (true, false) => sk.ground_top,
                        (false, true) => sk.ground_bottom,
                        (false, false) => sk.ground_fill,
                    };
                    Some(set.pick(l, r))
                }
                Tile::Platform => {
                    let p = |t: Tile| t == Tile::Platform;
                    Some(sk.platform.pick(is(xi - 1, yi, p), is(xi + 1, yi, p)))
                }
                Tile::Ladder => {
                    let top = level.get_or_empty(xi, yi - 1) != Tile::Ladder;
                    Some(if top { sk.ladder_top } else { sk.ladder })
                }
                Tile::Hazard => Some(variant(sk.hazard, x, y)),
                Tile::Coin => Some(variant(sk.coin, x, y)),
                Tile::Decor => Some(variant(sk.decor, x, y)),
                Tile::Enemy => {
                    // Personaje de 24 px centrado en la celda, apoyado en el suelo.
                    let off = i64::from((BIG - TILE) * s);
                    img.draw(sets.sprite(sk.enemy), px - off / 2, py - off, s);
                    None
                }
                Tile::Start => {
                    let off = i64::from((BIG - TILE) * s);
                    img.draw(
                        sets.sprite(Spr(Sheet::Characters, 0)),
                        px - off / 2,
                        py - off,
                        s,
                    );
                    None
                }
                Tile::Goal => {
                    img.draw(sets.sprite(Spr(Sheet::Base, 112)), px, py - t, s);
                    Some(Spr(Sheet::Base, 131))
                }
            };
            if let Some(sp) = spr {
                img.draw(sets.sprite(sp), px, py, s);
            }
        }
    }

    // Camino superpuesto.
    let center = |(x, y): (usize, usize)| (x as i64 * t + t / 2, y as i64 * t + t / 2);
    for pair in opts.path.windows(2) {
        img.line(
            center(pair[0]),
            center(pair[1]),
            i64::from(s.max(2)),
            [255, 230, 40, 220],
        );
    }
    img
}

/// Dibuja una tira de todos los sprites de una hoja con su índice (para inspección).
pub fn contact_sheet(sets: &Tilesets, style: Style, scale: u32) -> Image {
    let sheet = &sets.sheets[style.index()];
    let cols = 16_u32;
    let rows = (sheet.len() as u32).div_ceil(cols);
    let cell = TILE * scale + 2;
    let mut img = Image::new(cols * cell, rows * cell, [40, 40, 60, 255]);
    for (i, spr) in sheet.iter().enumerate() {
        let (c, r) = (i as u32 % cols, i as u32 / cols);
        img.draw(spr, i64::from(c * cell + 1), i64::from(r * cell + 1), scale);
    }
    img
}

impl Tilesets {
    /// Color representativo de un tile lógico en un estilo (color medio de su
    /// sprite), para miniaturas y para el visor de terminal.
    pub fn tile_color(&self, style: Style, tile: Tile) -> Rgba {
        let sk = skin(style);
        let spr = match tile {
            Tile::Empty => Spr(Sheet::Backgrounds, sk.background[0]),
            Tile::Ground => sk.ground_top.0[2],
            Tile::Platform => sk.platform.0[2],
            Tile::Ladder => sk.ladder,
            Tile::Hazard => sk.hazard[0],
            Tile::Coin => sk.coin[0],
            Tile::Decor => sk.decor[0],
            Tile::Enemy => sk.enemy,
            Tile::Start => Spr(Sheet::Characters, 0),
            Tile::Goal => Spr(Sheet::Base, 112),
        };
        mean_color(self.sprite(spr))
    }
}

/// Color medio no transparente de un sprite (útil para miniaturas).
pub fn mean_color(img: &Image) -> Rgba {
    let (mut sum, mut n) = ([0_u64; 3], 0_u64);
    for px in img.pixels().iter().filter(|p| p[3] > 0) {
        for c in 0..3 {
            sum[c] += u64::from(px[c]);
        }
        n += 1;
    }
    if n == 0 {
        return [0, 0, 0, 0];
    }
    [
        (sum[0] / n) as u8,
        (sum[1] / n) as u8,
        (sum[2] / n) as u8,
        255,
    ]
}
