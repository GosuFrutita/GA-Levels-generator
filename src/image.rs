//! Imágenes RGBA en memoria, composición y E/S en PNG hecha a mano.

pub mod checksum;
pub mod inflate;
pub mod png;

/// Píxel RGBA de 8 bits por canal.
pub type Rgba = [u8; 4];

/// Imagen RGBA en un búfer contiguo, fila por fila.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    width: u32,
    height: u32,
    pixels: Vec<Rgba>,
}

impl Image {
    /// Crea una imagen rellena con `fill`.
    pub fn new(width: u32, height: u32, fill: Rgba) -> Self {
        Self {
            width,
            height,
            pixels: vec![fill; width as usize * height as usize],
        }
    }

    /// Ancho en píxeles.
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Alto en píxeles.
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Píxeles fila por fila.
    pub fn pixels(&self) -> &[Rgba] {
        &self.pixels
    }

    #[inline]
    fn idx(&self, x: u32, y: u32) -> usize {
        debug_assert!(x < self.width && y < self.height);
        y as usize * self.width as usize + x as usize
    }

    /// Lee un píxel.
    ///
    /// # Panics
    ///
    /// Si `(x, y)` está fuera de la imagen.
    #[inline]
    pub fn get(&self, x: u32, y: u32) -> Rgba {
        self.pixels[self.idx(x, y)]
    }

    /// Escribe un píxel.
    ///
    /// # Panics
    ///
    /// Si `(x, y)` está fuera de la imagen.
    #[inline]
    pub fn set(&mut self, x: u32, y: u32, px: Rgba) {
        let i = self.idx(x, y);
        self.pixels[i] = px;
    }

    /// Mezcla `src` sobre el píxel `(x, y)` con el operador *over*; ignora
    /// coordenadas fuera de la imagen.
    #[inline]
    pub fn blend(&mut self, x: i64, y: i64, src: Rgba) {
        if x < 0 || y < 0 || x >= i64::from(self.width) || y >= i64::from(self.height) {
            return;
        }
        let i = self.idx(x as u32, y as u32);
        let dst = &mut self.pixels[i];
        match src[3] {
            0 => {}
            255 => *dst = src,
            a => {
                let a = u32::from(a);
                let inv = 255 - a;
                for c in 0..3 {
                    dst[c] = ((u32::from(src[c]) * a + u32::from(dst[c]) * inv + 127) / 255) as u8;
                }
                dst[3] = (a + u32::from(dst[3]) * inv / 255).min(255) as u8;
            }
        }
    }

    /// Recorta el rectángulo `(x, y, w, h)`.
    ///
    /// # Panics
    ///
    /// Si el rectángulo se sale de la imagen.
    #[must_use]
    pub fn crop(&self, x: u32, y: u32, w: u32, h: u32) -> Self {
        assert!(
            x + w <= self.width && y + h <= self.height,
            "recorte fuera de la imagen"
        );
        let mut out = Self::new(w, h, [0, 0, 0, 0]);
        for row in 0..h {
            let src = self.idx(x, y + row);
            let dst = (row * w) as usize;
            out.pixels[dst..dst + w as usize].copy_from_slice(&self.pixels[src..src + w as usize]);
        }
        out
    }

    /// Dibuja `src` escalado por `scale` (vecino más cercano) con mezcla alfa,
    /// con su esquina superior izquierda en `(x, y)`.
    pub fn draw(&mut self, src: &Self, x: i64, y: i64, scale: u32) {
        let s = i64::from(scale.max(1));
        for sy in 0..src.height {
            for sx in 0..src.width {
                let px = src.get(sx, sy);
                if px[3] == 0 {
                    continue;
                }
                for dy in 0..s {
                    for dx in 0..s {
                        self.blend(x + i64::from(sx) * s + dx, y + i64::from(sy) * s + dy, px);
                    }
                }
            }
        }
    }

    /// Rellena un rectángulo con mezcla alfa.
    pub fn fill_rect(&mut self, x: i64, y: i64, w: i64, h: i64, color: Rgba) {
        for yy in y..y + h {
            for xx in x..x + w {
                self.blend(xx, yy, color);
            }
        }
    }

    /// Dibuja una línea (Bresenham) de grosor `thick` píxeles.
    pub fn line(&mut self, (x0, y0): (i64, i64), (x1, y1): (i64, i64), thick: i64, color: Rgba) {
        let (dx, dy) = ((x1 - x0).abs(), -(y1 - y0).abs());
        let (sx, sy) = (if x0 < x1 { 1 } else { -1 }, if y0 < y1 { 1 } else { -1 });
        let (mut x, mut y, mut e) = (x0, y0, dx + dy);
        let r = thick / 2;
        loop {
            self.fill_rect(x - r, y - r, thick, thick, color);
            if x == x1 && y == y1 {
                break;
            }
            let e2 = 2 * e;
            if e2 >= dy {
                e += dy;
                x += sx;
            }
            if e2 <= dx {
                e += dx;
                y += sy;
            }
        }
    }
}
