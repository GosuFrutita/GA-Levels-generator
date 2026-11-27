//! Compresor DEFLATE (RFC 1951) con envoltorio zlib (RFC 1950).
//!
//! Busca repeticiones con LZ77 sobre una ventana de 32 KiB (cadenas de hash
//! de tres bytes, como zlib) y las codifica en un único bloque con los códigos
//! Huffman fijos del estándar. Los fondos de los niveles repiten mucho los
//! mismos tiles, así que basta para reducirlos a una fracción de su tamaño.

use super::checksum::adler32;
use super::inflate::{DIST_BASE, DIST_EXTRA, LEN_BASE, LEN_EXTRA};

const WINDOW: usize = 32_768;
const MIN_MATCH: usize = 3;
const MAX_MATCH: usize = 258;
/// Candidatos que se prueban por posición: más comprime algo mejor y tarda más.
const MAX_CHAIN: usize = 64;
const HASH_BITS: u32 = 15;
/// Marca de lista vacía en las cadenas de hash.
const NONE: usize = usize::MAX;

/// Escritor de bits LSB-primero.
#[derive(Default)]
struct BitWriter {
    out: Vec<u8>,
    acc: u64,
    bits: u32,
}

impl BitWriter {
    fn put(&mut self, value: u32, bits: u32) {
        self.acc |= u64::from(value) << self.bits;
        self.bits += bits;
        while self.bits >= 8 {
            self.out.push(self.acc as u8);
            self.acc >>= 8;
            self.bits -= 8;
        }
    }

    /// Los códigos Huffman se transmiten empezando por el bit más significativo.
    fn code(&mut self, code: u32, len: u32) {
        self.put(code.reverse_bits() >> (32 - len), len);
    }

    /// Símbolo del alfabeto de literales y longitudes con el código fijo.
    fn symbol(&mut self, s: u32) {
        let (code, len) = match s {
            0..=143 => (0x30 + s, 8),
            144..=255 => (0x190 + s - 144, 9),
            256..=279 => (s - 256, 7),
            _ => (0xC0 + s - 280, 8),
        };
        self.code(code, len);
    }

    fn copy(&mut self, len: usize, dist: usize) {
        let i = LEN_BASE.partition_point(|&b| usize::from(b) <= len) - 1;
        self.symbol(257 + i as u32);
        self.put(
            (len - usize::from(LEN_BASE[i])) as u32,
            u32::from(LEN_EXTRA[i]),
        );
        let d = DIST_BASE.partition_point(|&b| usize::from(b) <= dist) - 1;
        self.code(d as u32, 5);
        self.put(
            (dist - usize::from(DIST_BASE[d])) as u32,
            u32::from(DIST_EXTRA[d]),
        );
    }

    fn finish(mut self) -> Vec<u8> {
        if self.bits > 0 {
            self.out.push(self.acc as u8);
        }
        self.out
    }
}

fn hash(d: &[u8]) -> usize {
    let v = u32::from(d[0]) << 16 | u32::from(d[1]) << 8 | u32::from(d[2]);
    (v.wrapping_mul(2_654_435_761) >> (32 - HASH_BITS)) as usize
}

/// Comprime datos en un flujo zlib.
pub fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::default();
    // CMF = 0x78 (DEFLATE, ventana de 32 KiB) y FLG = 0x01 (múltiplo de 31).
    w.out.extend_from_slice(&[0x78, 0x01]);
    w.put(1, 1); // último bloque
    w.put(1, 2); // Huffman fijo
    let mut head = vec![NONE; 1 << HASH_BITS];
    let mut prev = vec![NONE; WINDOW];
    let mut i = 0;
    while i < data.len() {
        let max = (data.len() - i).min(MAX_MATCH);
        let (mut best_len, mut best_dist) = (0, 0);
        if max >= MIN_MATCH {
            let mut cand = head[hash(&data[i..])];
            for _ in 0..MAX_CHAIN {
                if cand == NONE || i - cand > WINDOW {
                    break;
                }
                let len = data[cand..]
                    .iter()
                    .zip(&data[i..i + max])
                    .take_while(|(a, b)| a == b)
                    .count();
                if len > best_len {
                    (best_len, best_dist) = (len, i - cand);
                    if len == max {
                        break;
                    }
                }
                // Una entrada reescrita por una posición más reciente corta la cadena.
                let next = prev[cand % WINDOW];
                if next == NONE || next >= cand {
                    break;
                }
                cand = next;
            }
        }
        let step = if best_len >= MIN_MATCH {
            w.copy(best_len, best_dist);
            best_len
        } else {
            w.symbol(u32::from(data[i]));
            1
        };
        for p in i..(i + step).min(data.len().saturating_sub(MIN_MATCH - 1)) {
            let h = hash(&data[p..]);
            prev[p % WINDOW] = head[h];
            head[h] = p;
        }
        i += step;
    }
    w.symbol(256);
    let mut out = w.finish();
    out.extend_from_slice(&adler32(data).to_be_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::inflate::zlib_decompress;

    #[test]
    fn ida_y_vuelta() {
        let mut x = 12_345_u32;
        let ruido: Vec<u8> = (0..70_000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                x as u8
            })
            .collect();
        let repetido: Vec<u8> = (0..200_000_u32).map(|i| (i % 300 / 7) as u8).collect();
        for data in [&[][..], b"a", b"abcabcabcabcabcabc", &ruido, &repetido] {
            let z = zlib_compress(data);
            assert_eq!(zlib_decompress(&z).expect("flujo válido"), data);
        }
        assert!(zlib_compress(&repetido).len() < repetido.len() / 20);
    }
}
