//! Lectura y escritura de PNG (W3C PNG, 3.ª edición).
//!
//! El decodificador soporta todos los tipos de color sin entrelazado, con
//! profundidades de 1 a 8 (y 16, reducida a 8 bits), paleta (`PLTE`) y
//! transparencia (`tRNS`). El codificador escribe RGBA8 comprimido con el
//! compresor DEFLATE propio ([`super::deflate`]).

use super::Image;
use super::checksum::Crc32;
use super::deflate::zlib_compress;
use super::inflate::zlib_decompress;
use crate::error::{Error, Result};
use std::path::Path;

const SIGNATURE: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];

fn err(detail: impl Into<String>) -> Error {
    Error::format("PNG", detail)
}

#[derive(Debug, Clone, Copy)]
struct Header {
    width: u32,
    height: u32,
    bit_depth: u8,
    color_type: u8,
}

impl Header {
    fn channels(self) -> Result<usize> {
        let ok_depths: &[u8] = match self.color_type {
            0 => &[1, 2, 4, 8, 16],
            3 => &[1, 2, 4, 8],
            2 | 4 | 6 => &[8, 16],
            _ => {
                return Err(err(format!(
                    "tipo de color {} desconocido",
                    self.color_type
                )));
            }
        };
        if !ok_depths.contains(&self.bit_depth) {
            return Err(err(format!(
                "profundidad {} inválida para el tipo de color {}",
                self.bit_depth, self.color_type
            )));
        }
        Ok(match self.color_type {
            0 | 3 => 1,
            2 => 3,
            4 => 2,
            _ => 4,
        })
    }
}

fn be_u32(b: &[u8]) -> u32 {
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// Predictor de Paeth (sección 9.4 de la especificación).
#[inline]
fn paeth(a: u8, b: u8, c: u8) -> u8 {
    let p = i16::from(a) + i16::from(b) - i16::from(c);
    let pa = (p - i16::from(a)).abs();
    let pb = (p - i16::from(b)).abs();
    let pc = (p - i16::from(c)).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Deshace los filtros de cada línea en sitio y devuelve las líneas sin el byte de filtro.
fn unfilter(raw: &[u8], stride: usize, height: usize, bpp: usize) -> Result<Vec<u8>> {
    if raw.len() < (stride + 1) * height {
        return Err(err("datos de imagen truncados"));
    }
    let mut out = vec![0_u8; stride * height];
    for y in 0..height {
        let line = &raw[y * (stride + 1)..(y + 1) * (stride + 1)];
        let (filter, src) = (line[0], &line[1..]);
        let (prev_rows, cur_rows) = out.split_at_mut(y * stride);
        let prev = if y == 0 {
            None
        } else {
            Some(&prev_rows[(y - 1) * stride..])
        };
        let cur = &mut cur_rows[..stride];
        for i in 0..stride {
            let a = if i >= bpp { cur[i - bpp] } else { 0 };
            let b = prev.map_or(0, |p| p[i]);
            let c = if i >= bpp {
                prev.map_or(0, |p| p[i - bpp])
            } else {
                0
            };
            let pred = match filter {
                0 => 0,
                1 => a,
                2 => b,
                3 => u8::midpoint(a, b),
                4 => paeth(a, b, c),
                f => return Err(err(format!("filtro {f} desconocido"))),
            };
            cur[i] = src[i].wrapping_add(pred);
        }
    }
    Ok(out)
}

/// Extrae la muestra `index` de una línea con `depth` bits por muestra (MSB primero).
#[inline]
fn sample(line: &[u8], index: usize, depth: u8) -> u16 {
    match depth {
        8 => u16::from(line[index]),
        16 => u16::from_be_bytes([line[2 * index], line[2 * index + 1]]),
        d => {
            let bit = index * usize::from(d);
            let byte = line[bit / 8];
            let shift = 8 - usize::from(d) - bit % 8;
            u16::from((byte >> shift) & ((1_u8 << d) - 1))
        }
    }
}

/// Escala una muestra de `depth` bits a 8 bits.
#[inline]
fn to_u8(v: u16, depth: u8) -> u8 {
    match depth {
        16 => (v >> 8) as u8,
        8 => v as u8,
        d => (u32::from(v) * 255 / ((1_u32 << d) - 1)) as u8,
    }
}

/// Decodifica un PNG desde memoria.
///
/// # Errors
///
/// Si el archivo no es un PNG válido o usa entrelazado (no soportado).
pub fn decode(bytes: &[u8]) -> Result<Image> {
    if bytes.len() < 8 || bytes[..8] != SIGNATURE {
        return Err(err("firma incorrecta"));
    }
    let mut pos = 8;
    let mut header: Option<Header> = None;
    let mut palette: Vec<[u8; 4]> = Vec::new();
    let mut trns: Vec<u8> = Vec::new();
    let mut idat: Vec<u8> = Vec::new();
    loop {
        let head = bytes
            .get(pos..pos + 8)
            .ok_or_else(|| err("chunk truncado"))?;
        let len = be_u32(&head[..4]) as usize;
        let kind = [head[4], head[5], head[6], head[7]];
        let body = bytes
            .get(pos + 8..pos + 8 + len)
            .ok_or_else(|| err("chunk truncado"))?;
        let crc_bytes = bytes
            .get(pos + 8 + len..pos + 12 + len)
            .ok_or_else(|| err("falta CRC"))?;
        let mut crc = Crc32::new();
        crc.update(&kind);
        crc.update(body);
        if crc.finish() != be_u32(crc_bytes) {
            return Err(err(format!(
                "CRC incorrecto en {}",
                String::from_utf8_lossy(&kind)
            )));
        }
        pos += 12 + len;
        match &kind {
            b"IHDR" => {
                if len != 13 {
                    return Err(err("IHDR de longitud incorrecta"));
                }
                if body[10] != 0 || body[11] != 0 {
                    return Err(err("método de compresión o filtro desconocido"));
                }
                if body[12] != 0 {
                    return Err(err("PNG entrelazado (Adam7) no soportado"));
                }
                header = Some(Header {
                    width: be_u32(&body[0..4]),
                    height: be_u32(&body[4..8]),
                    bit_depth: body[8],
                    color_type: body[9],
                });
            }
            b"PLTE" => {
                if !len.is_multiple_of(3) || len / 3 > 256 {
                    return Err(err("PLTE inválido"));
                }
                palette = body
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(|&[r, g, b]| [r, g, b, 255])
                    .collect();
            }
            b"tRNS" => trns = body.to_vec(),
            b"IDAT" => idat.extend_from_slice(body),
            b"IEND" => break,
            _ => {} // Chunks auxiliares (gAMA, pHYs, ...): se ignoran.
        }
    }
    let h = header.ok_or_else(|| err("falta IHDR"))?;
    let channels = h.channels()?;
    let (w, ht) = (h.width as usize, h.height as usize);
    if w == 0 || ht == 0 {
        return Err(err("dimensiones nulas"));
    }
    let bits_per_pixel = channels * usize::from(h.bit_depth);
    let stride = (w * bits_per_pixel).div_ceil(8);
    let bpp = bits_per_pixel.div_ceil(8).max(1);
    let raw = zlib_decompress(&idat)?;
    let data = unfilter(&raw, stride, ht, bpp)?;

    if h.color_type == 3 {
        if palette.is_empty() {
            return Err(err("imagen indexada sin PLTE"));
        }
        for (entry, &alpha) in palette.iter_mut().zip(&trns) {
            entry[3] = alpha;
        }
    }
    // Color clave de tRNS para escala de grises / RGB (valores en la profundidad original).
    let key: Option<[u16; 3]> = match (h.color_type, trns.len()) {
        (0, 2) => {
            let g = u16::from_be_bytes([trns[0], trns[1]]);
            Some([g, g, g])
        }
        (2, 6) => Some([
            u16::from_be_bytes([trns[0], trns[1]]),
            u16::from_be_bytes([trns[2], trns[3]]),
            u16::from_be_bytes([trns[4], trns[5]]),
        ]),
        _ => None,
    };

    expand(h, &data, stride, &palette, key)
}

/// Convierte las líneas ya desfiltradas a RGBA8.
fn expand(
    h: Header,
    data: &[u8],
    stride: usize,
    palette: &[[u8; 4]],
    key: Option<[u16; 3]>,
) -> Result<Image> {
    let (w, ht) = (h.width as usize, h.height as usize);
    let mut img = Image::new(h.width, h.height, [0, 0, 0, 0]);
    let d = h.bit_depth;
    for y in 0..ht {
        let line = &data[y * stride..(y + 1) * stride];
        for x in 0..w {
            let px = match h.color_type {
                0 => {
                    let g = sample(line, x, d);
                    let a = if key.is_some_and(|k| k[0] == g) {
                        0
                    } else {
                        255
                    };
                    let v = to_u8(g, d);
                    [v, v, v, a]
                }
                2 => {
                    let s = [
                        sample(line, 3 * x, d),
                        sample(line, 3 * x + 1, d),
                        sample(line, 3 * x + 2, d),
                    ];
                    let a = if key == Some(s) { 0 } else { 255 };
                    [to_u8(s[0], d), to_u8(s[1], d), to_u8(s[2], d), a]
                }
                3 => {
                    let i = usize::from(sample(line, x, d));
                    *palette
                        .get(i)
                        .ok_or_else(|| err("índice fuera de la paleta"))?
                }
                4 => {
                    let v = to_u8(sample(line, 2 * x, d), d);
                    [v, v, v, to_u8(sample(line, 2 * x + 1, d), d)]
                }
                _ => [
                    to_u8(sample(line, 4 * x, d), d),
                    to_u8(sample(line, 4 * x + 1, d), d),
                    to_u8(sample(line, 4 * x + 2, d), d),
                    to_u8(sample(line, 4 * x + 3, d), d),
                ],
            };
            img.set(x as u32, y as u32, px);
        }
    }
    Ok(img)
}

/// Lee y decodifica un archivo PNG.
///
/// # Errors
///
/// Si el archivo no se puede leer o no es un PNG válido.
pub fn load(path: impl AsRef<Path>) -> Result<Image> {
    decode(&std::fs::read(path)?)
}

fn write_chunk(out: &mut Vec<u8>, kind: [u8; 4], body: &[u8]) {
    out.extend_from_slice(&(body.len() as u32).to_be_bytes());
    out.extend_from_slice(&kind);
    out.extend_from_slice(body);
    let mut crc = Crc32::new();
    crc.update(&kind);
    crc.update(body);
    out.extend_from_slice(&crc.finish().to_be_bytes());
}

/// Codifica una imagen como PNG RGBA de 8 bits.
pub fn encode(img: &Image) -> Vec<u8> {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut raw = Vec::with_capacity((w * 4 + 1) * h);
    for row in img.pixels().chunks_exact(w) {
        raw.push(0); // filtro None
        for px in row {
            raw.extend_from_slice(px);
        }
    }
    let z = zlib_compress(&raw);
    let mut out = Vec::with_capacity(z.len() + 64);
    out.extend_from_slice(&SIGNATURE);
    let mut ihdr = [0_u8; 13];
    ihdr[..4].copy_from_slice(&img.width().to_be_bytes());
    ihdr[4..8].copy_from_slice(&img.height().to_be_bytes());
    ihdr[8] = 8; // profundidad
    ihdr[9] = 6; // RGBA
    write_chunk(&mut out, *b"IHDR", &ihdr);
    write_chunk(&mut out, *b"IDAT", &z);
    write_chunk(&mut out, *b"IEND", &[]);
    out
}

/// Codifica y guarda una imagen PNG.
///
/// # Errors
///
/// Si no se puede escribir el archivo.
pub fn save(img: &Image, path: impl AsRef<Path>) -> Result<()> {
    std::fs::write(path, encode(img))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ida_y_vuelta() {
        let mut img = Image::new(37, 11, [0, 0, 0, 0]);
        for y in 0..11 {
            for x in 0..37 {
                img.set(x, y, [(x * 7) as u8, (y * 23) as u8, (x ^ y) as u8, 200]);
            }
        }
        let back = decode(&encode(&img)).expect("PNG válido");
        assert_eq!(back, img);
    }

    #[test]
    fn paeth_casos() {
        assert_eq!(paeth(10, 20, 10), 20);
        assert_eq!(paeth(20, 10, 10), 20);
        assert_eq!(paeth(5, 5, 5), 5);
    }

    #[test]
    fn rechaza_firma() {
        assert!(decode(b"no es un png").is_err());
    }
}
