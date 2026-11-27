//! Sumas de verificación de PNG (CRC-32) y zlib (Adler-32).

/// Tabla CRC-32 (polinomio reflejado `0xEDB88320`), calculada en compilación.
const CRC_TABLE: [u32; 256] = {
    let mut table = [0_u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                0xEDB8_8320 ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
};

/// CRC-32 incremental, tal como lo define la especificación PNG.
#[derive(Debug, Clone, Copy)]
pub struct Crc32(u32);

impl Crc32 {
    /// Estado inicial.
    pub const fn new() -> Self {
        Self(0xFFFF_FFFF)
    }

    /// Procesa más bytes.
    pub fn update(&mut self, bytes: &[u8]) {
        let mut c = self.0;
        for &b in bytes {
            c = CRC_TABLE[((c ^ u32::from(b)) & 0xFF) as usize] ^ (c >> 8);
        }
        self.0 = c;
    }

    /// Valor final.
    pub const fn finish(self) -> u32 {
        self.0 ^ 0xFFFF_FFFF
    }
}

impl Default for Crc32 {
    fn default() -> Self {
        Self::new()
    }
}

/// Adler-32 (RFC 1950).
pub fn adler32(bytes: &[u8]) -> u32 {
    const MOD: u32 = 65_521;
    // 5552 es el mayor bloque que no desborda u32 antes de reducir (zlib).
    let (mut a, mut b) = (1_u32, 0_u32);
    for chunk in bytes.chunks(5552) {
        for &x in chunk {
            a += u32::from(x);
            b += a;
        }
        a %= MOD;
        b %= MOD;
    }
    (b << 16) | a
}

#[cfg(test)]
mod tests {
    use super::*;

    fn crc32(bytes: &[u8]) -> u32 {
        let mut c = Crc32::new();
        c.update(bytes);
        c.finish()
    }

    #[test]
    fn crc_de_iend() {
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
    }

    #[test]
    fn crc_vector_estandar() {
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn adler_vector_estandar() {
        assert_eq!(adler32(b"Wikipedia"), 0x11E6_0398);
    }
}
