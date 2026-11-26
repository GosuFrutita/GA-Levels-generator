//! Descompresor DEFLATE (RFC 1951) y envoltorio zlib (RFC 1950).
//!
//! Implementa los tres tipos de bloque: almacenado, Huffman fijo y Huffman
//! dinámico. La decodificación Huffman sigue el esquema canónico de `puff.c`
//! de Mark Adler: recuento de códigos por longitud y símbolos ordenados.

use super::checksum::adler32;
use crate::error::{Error, Result};

const MAX_BITS: usize = 15;

/// Longitud base de los símbolos 257..=285.
const LEN_BASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
/// Bits extra de los símbolos de longitud.
const LEN_EXTRA: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
/// Distancia base de los códigos 0..=29.
const DIST_BASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
/// Bits extra de las distancias.
const DIST_EXTRA: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];
/// Orden en que se transmiten las longitudes del alfabeto de longitudes.
const CL_ORDER: [usize; 19] = [
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
];

fn err(detail: &str) -> Error {
    Error::format("flujo DEFLATE", detail)
}

/// Lector de bits LSB-primero.
struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    buf: u64,
    count: u32,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            pos: 0,
            buf: 0,
            count: 0,
        }
    }

    fn bits(&mut self, n: u32) -> Result<u32> {
        debug_assert!(n <= 32);
        while self.count < n {
            let byte = *self
                .data
                .get(self.pos)
                .ok_or_else(|| err("datos truncados"))?;
            self.pos += 1;
            self.buf |= u64::from(byte) << self.count;
            self.count += 8;
        }
        let v = (self.buf & ((1_u64 << n) - 1)) as u32;
        self.buf >>= n;
        self.count -= n;
        Ok(v)
    }

    /// Descarta los bits restantes del byte actual.
    fn align(&mut self) {
        let drop = self.count % 8;
        self.buf >>= drop;
        self.count -= drop;
    }

    fn read_aligned_u16_le(&mut self) -> Result<u16> {
        let lo = self.bits(8)?;
        let hi = self.bits(8)?;
        Ok((lo | (hi << 8)) as u16)
    }

    /// Copia `n` bytes alineados (bloque almacenado).
    fn copy_aligned(&mut self, n: usize, out: &mut Vec<u8>) -> Result<()> {
        // Primero vacía los bytes que ya están en el búfer de bits.
        let mut remaining = n;
        while remaining > 0 && self.count >= 8 {
            out.push(self.bits(8)? as u8);
            remaining -= 1;
        }
        let end = self
            .pos
            .checked_add(remaining)
            .ok_or_else(|| err("longitud inválida"))?;
        let slice = self
            .data
            .get(self.pos..end)
            .ok_or_else(|| err("bloque almacenado truncado"))?;
        out.extend_from_slice(slice);
        self.pos = end;
        Ok(())
    }
}

/// Código Huffman canónico.
struct Huffman {
    counts: [u16; MAX_BITS + 1],
    symbols: Vec<u16>,
}

impl Huffman {
    fn new(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0_u16; MAX_BITS + 1];
        for &l in lengths {
            counts[usize::from(l)] += 1;
        }
        counts[0] = 0;
        // Comprueba que el código no esté sobresuscrito.
        let mut left = 1_i32;
        for &c in &counts[1..] {
            left = (left << 1) - i32::from(c);
            if left < 0 {
                return Err(err("código Huffman sobresuscrito"));
            }
        }
        let mut offsets = [0_u16; MAX_BITS + 2];
        for len in 1..=MAX_BITS {
            offsets[len + 1] = offsets[len] + counts[len];
        }
        let mut symbols = vec![0_u16; lengths.len()];
        for (sym, &l) in lengths.iter().enumerate() {
            if l != 0 {
                let slot = &mut offsets[usize::from(l)];
                symbols[usize::from(*slot)] = sym as u16;
                *slot += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, br: &mut BitReader<'_>) -> Result<u16> {
        let (mut code, mut first, mut index) = (0_i32, 0_i32, 0_i32);
        for len in 1..=MAX_BITS {
            code |= br.bits(1)?.cast_signed();
            let count = i32::from(self.counts[len]);
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first += count;
            first <<= 1;
            code <<= 1;
        }
        Err(err("código Huffman inválido"))
    }
}

fn fixed_tables() -> Result<(Huffman, Huffman)> {
    let mut lit = [0_u8; 288];
    lit[..144].fill(8);
    lit[144..256].fill(9);
    lit[256..280].fill(7);
    lit[280..].fill(8);
    let dist = [5_u8; 30];
    Ok((Huffman::new(&lit)?, Huffman::new(&dist)?))
}

fn dynamic_tables(br: &mut BitReader<'_>) -> Result<(Huffman, Huffman)> {
    let hlit = br.bits(5)? as usize + 257;
    let hdist = br.bits(5)? as usize + 1;
    let hclen = br.bits(4)? as usize + 4;
    if hlit > 286 || hdist > 30 {
        return Err(err("demasiados códigos"));
    }
    let mut cl_lengths = [0_u8; 19];
    for &i in &CL_ORDER[..hclen] {
        cl_lengths[i] = br.bits(3)? as u8;
    }
    let cl = Huffman::new(&cl_lengths)?;

    let mut lengths = [0_u8; 286 + 30];
    let total = hlit + hdist;
    let mut i = 0;
    while i < total {
        let sym = cl.decode(br)?;
        let (value, repeat) = match sym {
            0..=15 => (sym as u8, 1),
            16 => {
                let prev = *lengths[..i]
                    .last()
                    .ok_or_else(|| err("repetición sin longitud previa"))?;
                (prev, 3 + br.bits(2)? as usize)
            }
            17 => (0, 3 + br.bits(3)? as usize),
            18 => (0, 11 + br.bits(7)? as usize),
            _ => return Err(err("símbolo de longitud inválido")),
        };
        if i + repeat > total {
            return Err(err("demasiadas longitudes"));
        }
        lengths[i..i + repeat].fill(value);
        i += repeat;
    }
    if lengths[256] == 0 {
        return Err(err("falta el código de fin de bloque"));
    }
    Ok((
        Huffman::new(&lengths[..hlit])?,
        Huffman::new(&lengths[hlit..total])?,
    ))
}

fn inflate_block(
    br: &mut BitReader<'_>,
    lit: &Huffman,
    dist: &Huffman,
    out: &mut Vec<u8>,
) -> Result<()> {
    loop {
        let sym = lit.decode(br)?;
        match sym {
            0..=255 => out.push(sym as u8),
            256 => return Ok(()),
            257..=285 => {
                let li = usize::from(sym - 257);
                let len = usize::from(LEN_BASE[li]) + br.bits(u32::from(LEN_EXTRA[li]))? as usize;
                let di = usize::from(dist.decode(br)?);
                if di >= 30 {
                    return Err(err("código de distancia inválido"));
                }
                let d = usize::from(DIST_BASE[di]) + br.bits(u32::from(DIST_EXTRA[di]))? as usize;
                if d > out.len() {
                    return Err(err("distancia hacia atrás fuera del búfer"));
                }
                // La copia puede solaparse (d < len), así que va byte a byte.
                let start = out.len() - d;
                for k in 0..len {
                    let b = out[start + k];
                    out.push(b);
                }
            }
            _ => return Err(err("símbolo literal/longitud inválido")),
        }
    }
}

/// Descomprime un flujo DEFLATE crudo. Devuelve los datos y los bytes consumidos.
///
/// # Errors
///
/// Si el flujo está truncado o es inválido.
pub fn inflate(data: &[u8]) -> Result<(Vec<u8>, usize)> {
    let mut br = BitReader::new(data);
    let mut out = Vec::with_capacity(data.len() * 4);
    let mut fixed: Option<(Huffman, Huffman)> = None;
    loop {
        let last = br.bits(1)? == 1;
        match br.bits(2)? {
            0 => {
                br.align();
                let len = br.read_aligned_u16_le()?;
                let nlen = br.read_aligned_u16_le()?;
                if len != !nlen {
                    return Err(err("LEN/NLEN no coinciden"));
                }
                br.copy_aligned(usize::from(len), &mut out)?;
            }
            1 => {
                let tables = match fixed.take() {
                    Some(t) => t,
                    None => fixed_tables()?,
                };
                inflate_block(&mut br, &tables.0, &tables.1, &mut out)?;
                fixed = Some(tables);
            }
            2 => {
                let (lit, dist) = dynamic_tables(&mut br)?;
                inflate_block(&mut br, &lit, &dist, &mut out)?;
            }
            _ => return Err(err("tipo de bloque reservado")),
        }
        if last {
            break;
        }
    }
    br.align();
    let consumed = br.pos - (br.count / 8) as usize;
    Ok((out, consumed))
}

/// Descomprime un flujo zlib y verifica su Adler-32.
///
/// # Errors
///
/// Si la cabecera, el flujo o la suma de verificación son inválidos.
pub fn zlib_decompress(data: &[u8]) -> Result<Vec<u8>> {
    let zerr = |d: &str| Error::format("flujo zlib", d);
    let [cmf, flg, ..] = *data else {
        return Err(zerr("demasiado corto"));
    };
    if cmf & 0x0F != 8 {
        return Err(zerr("método de compresión distinto de DEFLATE"));
    }
    if (u16::from(cmf) << 8 | u16::from(flg)) % 31 != 0 {
        return Err(zerr("FCHECK incorrecto"));
    }
    if flg & 0x20 != 0 {
        return Err(zerr("diccionario predefinido no soportado"));
    }
    let (out, used) = inflate(&data[2..])?;
    let tail = data
        .get(2 + used..2 + used + 4)
        .ok_or_else(|| zerr("falta Adler-32"))?;
    let expected = u32::from_be_bytes([tail[0], tail[1], tail[2], tail[3]]);
    if adler32(&out) != expected {
        return Err(zerr("Adler-32 no coincide"));
    }
    Ok(out)
}

/// Envuelve datos en un flujo zlib con bloques DEFLATE almacenados (sin comprimir).
pub fn zlib_store(data: &[u8]) -> Vec<u8> {
    const MAX_BLOCK: usize = 65_535;
    let blocks = data.len().div_ceil(MAX_BLOCK).max(1);
    let mut out = Vec::with_capacity(data.len() + blocks * 5 + 6);
    out.extend_from_slice(&[0x78, 0x01]);
    let mut chunks = data.chunks(MAX_BLOCK).peekable();
    if chunks.peek().is_none() {
        out.extend_from_slice(&[1, 0, 0, 0xFF, 0xFF]);
    }
    while let Some(chunk) = chunks.next() {
        let last = u8::from(chunks.peek().is_none());
        let len = chunk.len() as u16;
        out.push(last);
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(chunk);
    }
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ida_y_vuelta_almacenado() {
        let data: Vec<u8> = (0..200_000_u32).map(|i| (i * 7 % 251) as u8).collect();
        assert_eq!(zlib_decompress(&zlib_store(&data)).expect("válido"), data);
        assert_eq!(
            zlib_decompress(&zlib_store(&[])).expect("válido"),
            Vec::<u8>::new()
        );
    }

    #[test]
    fn huffman_fijo_de_zlib() {
        // zlib.compress(b"hello hello hello hello", 9): bloque con Huffman fijo.
        let z = [
            0x78, 0xda, 0xcb, 0x48, 0xcd, 0xc9, 0xc9, 0x57, 0xc8, 0x40, 0x27, 0x01, 0x68, 0x03,
            0x08, 0xb1,
        ];
        assert_eq!(
            zlib_decompress(&z).expect("válido"),
            b"hello hello hello hello"
        );
    }

    #[test]
    fn rechaza_basura() {
        assert!(zlib_decompress(&[0x78, 0x9c, 0xff, 0xff]).is_err());
        assert!(zlib_decompress(&[0x00]).is_err());
    }
}
