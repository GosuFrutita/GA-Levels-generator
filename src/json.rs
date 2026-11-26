//! Escritura de JSON sin dependencias (RFC 8259): un constructor mínimo de
//! objetos y listas con escape correcto de cadenas.

use std::fmt::Write as _;

/// Valor JSON.
#[derive(Debug, Clone, PartialEq)]
pub enum Json {
    /// `null`.
    Null,
    /// Booleano.
    Bool(bool),
    /// Número (los no finitos se escriben como `null`).
    Num(f64),
    /// Cadena.
    Str(String),
    /// Lista.
    Arr(Vec<Json>),
    /// Objeto con el orden de inserción.
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Objeto vacío.
    pub fn obj() -> Self {
        Self::Obj(Vec::new())
    }

    /// Añade un campo a un objeto (encadenable).
    ///
    /// # Panics
    ///
    /// Si `self` no es un objeto.
    #[must_use]
    pub fn with(mut self, key: &str, value: impl Into<Self>) -> Self {
        match &mut self {
            Self::Obj(fields) => fields.push((key.to_string(), value.into())),
            _ => panic!("`with` solo se aplica a objetos"),
        }
        self
    }

    /// Serializa el valor.
    pub fn dump(&self) -> String {
        let mut s = String::new();
        self.write(&mut s);
        s
    }

    fn write(&self, s: &mut String) {
        match self {
            Self::Bool(b) => s.push_str(if *b { "true" } else { "false" }),
            Self::Num(n) if n.is_finite() => {
                if n.fract() == 0.0 && n.abs() < 1e15 {
                    let _ = write!(s, "{n:.0}");
                } else {
                    let _ = write!(s, "{n}");
                }
            }
            Self::Null | Self::Num(_) => s.push_str("null"),
            Self::Str(t) => escape(t, s),
            Self::Arr(v) => {
                s.push('[');
                for (i, x) in v.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    x.write(s);
                }
                s.push(']');
            }
            Self::Obj(fields) => {
                s.push('{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        s.push(',');
                    }
                    escape(k, s);
                    s.push(':');
                    v.write(s);
                }
                s.push('}');
            }
        }
    }
}

/// Escribe una cadena JSON entre comillas con sus escapes.
fn escape(t: &str, s: &mut String) {
    s.push('"');
    for c in t.chars() {
        match c {
            '"' => s.push_str("\\\""),
            '\\' => s.push_str("\\\\"),
            '\n' => s.push_str("\\n"),
            '\r' => s.push_str("\\r"),
            '\t' => s.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(s, "\\u{:04x}", c as u32);
            }
            c => s.push(c),
        }
    }
    s.push('"');
}

impl From<bool> for Json {
    fn from(b: bool) -> Self {
        Self::Bool(b)
    }
}

impl From<f64> for Json {
    fn from(n: f64) -> Self {
        Self::Num(n)
    }
}

impl From<u32> for Json {
    fn from(n: u32) -> Self {
        Self::Num(f64::from(n))
    }
}

impl From<usize> for Json {
    fn from(n: usize) -> Self {
        Self::Num(n as f64)
    }
}

impl From<u64> for Json {
    fn from(n: u64) -> Self {
        Self::Num(n as f64)
    }
}

impl From<&str> for Json {
    fn from(t: &str) -> Self {
        Self::Str(t.to_string())
    }
}

impl From<String> for Json {
    fn from(t: String) -> Self {
        Self::Str(t)
    }
}

impl<T: Into<Self>> From<Vec<T>> for Json {
    fn from(v: Vec<T>) -> Self {
        Self::Arr(v.into_iter().map(Into::into).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serializa_y_escapa() {
        let j = Json::obj()
            .with("a", 1_u32)
            .with("b", 0.5)
            .with("c", "x\"y\\z\n")
            .with("d", vec![true, false])
            .with("e", f64::NAN)
            .with("f", Json::obj());
        assert_eq!(
            j.dump(),
            r#"{"a":1,"b":0.5,"c":"x\"y\\z\n","d":[true,false],"e":null,"f":{}}"#
        );
    }
}
