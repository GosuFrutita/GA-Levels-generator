//! Análisis de argumentos de línea de órdenes sin dependencias.
//!
//! Formato: `nivelga <orden> [posicionales...] [--clave valor | --bandera]`.
//! Las opciones no consumidas se rechazan para detectar erratas.

use crate::error::{Error, Result};
use crate::fitness::Aggregation;
use crate::ga::selection::{Scaling, Selection};
use crate::ga::variation::Crossover;
use crate::ga::{GaConfig, OperatorControl};
use std::collections::BTreeMap;
use std::str::FromStr;

/// Argumentos analizados.
#[derive(Debug, Clone, Default)]
pub struct Args {
    /// Orden (primer argumento).
    pub command: String,
    /// Argumentos posicionales tras la orden.
    pub positional: Vec<String>,
    opts: BTreeMap<String, String>,
}

impl Args {
    /// Analiza una secuencia de argumentos (sin el nombre del programa).
    ///
    /// # Errors
    ///
    /// Si una opción aparece dos veces.
    pub fn parse(args: impl IntoIterator<Item = String>) -> Result<Self> {
        let mut it = args.into_iter().peekable();
        let command = it.next().unwrap_or_default();
        let mut out = Self {
            command,
            ..Self::default()
        };
        while let Some(a) = it.next() {
            if let Some(key) = a.strip_prefix("--") {
                let (k, v) = if let Some((k, v)) = key.split_once('=') {
                    (k.to_string(), v.to_string())
                } else {
                    let v = if it.peek().is_some_and(|n| !n.starts_with("--")) {
                        it.next().unwrap_or_default()
                    } else {
                        "si".to_string()
                    };
                    (key.to_string(), v)
                };
                if out.opts.insert(k.clone(), v).is_some() {
                    return Err(Error::Config(format!("opción --{k} repetida")));
                }
            } else {
                out.positional.push(a);
            }
        }
        Ok(out)
    }

    /// Extrae una opción como texto.
    pub fn take(&mut self, key: &str) -> Option<String> {
        self.opts.remove(key)
    }

    /// Extrae y convierte una opción, o devuelve `default`.
    ///
    /// # Errors
    ///
    /// Si el valor no se puede convertir.
    pub fn get<T: FromStr>(&mut self, key: &str, default: T) -> Result<T> {
        match self.opts.remove(key) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| Error::Config(format!("valor inválido para --{key}: «{v}»"))),
        }
    }

    /// Extrae una bandera booleana (`si`/`no`, `true`/`false`, `1`/`0`).
    ///
    /// # Errors
    ///
    /// Si el valor no es booleano.
    pub fn flag(&mut self, key: &str, default: bool) -> Result<bool> {
        match self.opts.remove(key).as_deref() {
            None => Ok(default),
            Some("si" | "sí" | "true" | "1") => Ok(true),
            Some("no" | "false" | "0") => Ok(false),
            Some(v) => Err(Error::Config(format!("--{key} espera si/no, no «{v}»"))),
        }
    }

    /// Falla si quedan opciones sin consumir.
    ///
    /// # Errors
    ///
    /// Si hay opciones desconocidas.
    pub fn finish(&self) -> Result<()> {
        if let Some(k) = self.opts.keys().next() {
            return Err(Error::Config(format!("opción desconocida --{k}")));
        }
        Ok(())
    }
}

fn num<T: FromStr>(s: &str, what: &str) -> Result<T> {
    s.parse()
        .map_err(|_| Error::Config(format!("número inválido en {what}: «{s}»")))
}

/// `torneo:K` · `ruleta[:cruda|:lineal:C|:sigma:C][:sus]`.
///
/// # Errors
///
/// Si el texto no describe una selección válida.
pub fn parse_selection(s: &str) -> Result<Selection> {
    let parts: Vec<&str> = s.split(':').collect();
    match parts.as_slice() {
        ["torneo"] => Ok(Selection::Tournament(3)),
        ["torneo", k] => Ok(Selection::Tournament(num(k, "torneo")?)),
        ["ruleta", rest @ ..] => {
            let sus = rest.last() == Some(&"sus");
            let rest = if sus { &rest[..rest.len() - 1] } else { rest };
            let scaling = match rest {
                [] | ["cruda"] => Scaling::None,
                ["lineal"] => Scaling::Linear(1.5),
                ["lineal", c] => Scaling::Linear(num(c, "escalado lineal")?),
                ["sigma"] => Scaling::Sigma(2.0),
                ["sigma", c] => Scaling::Sigma(num(c, "truncamiento sigma")?),
                _ => {
                    return Err(Error::Config(format!(
                        "escalado de ruleta desconocido: «{s}»"
                    )));
                }
            };
            Ok(Selection::Roulette { scaling, sus })
        }
        _ => Err(Error::Config(format!("selección desconocida: «{s}»"))),
    }
}

/// `columnas` · `bloque` · `uniforme` · `ninguno`.
///
/// # Errors
///
/// Si el nombre no es válido.
pub fn parse_crossover(s: &str) -> Result<Crossover> {
    [
        Crossover::Columns,
        Crossover::Block,
        Crossover::Uniform,
        Crossover::None,
    ]
    .into_iter()
    .find(|c| c.name() == s)
    .ok_or_else(|| Error::Config(format!("cruce desconocido: «{s}»")))
}

/// `suma` · `geometrica` · `penalizacion:λ` · `deb`.
///
/// # Errors
///
/// Si el texto no es válido.
pub fn parse_aggregation(s: &str) -> Result<Aggregation> {
    match s.split_once(':') {
        None if s == "suma" => Ok(Aggregation::WeightedSum),
        None if s == "geometrica" => Ok(Aggregation::Geometric),
        None if s == "deb" => Ok(Aggregation::Deb),
        None if s == "penalizacion" => Ok(Aggregation::Penalty(1.0)),
        Some(("penalizacion", l)) => Ok(Aggregation::Penalty(num(l, "penalización")?)),
        _ => Err(Error::Config(format!("agregación desconocida: «{s}»"))),
    }
}

/// Construye la configuración del GA a partir de las opciones.
///
/// # Errors
///
/// Si alguna opción es inválida.
pub fn ga_config(args: &mut Args) -> Result<GaConfig> {
    let d = GaConfig::default();
    let mut cfg = GaConfig {
        seed: args.get("semilla", d.seed)?,
        population: args.get("poblacion", d.population)?,
        generations: args.get("generaciones", d.generations)?,
        width: args.get("ancho", d.width)?,
        p_crossover: args.get("pc", d.p_crossover)?,
        mutations_per_child: args.get("mutaciones", d.mutations_per_child)?,
        p_macro: args.get("macro", d.p_macro)?,
        p_style: args.get("p-estilo", d.p_style)?,
        elitism: args.get("elitismo", d.elitism)?,
        repair: args.flag("reparar", d.repair)?,
        threads: args.get("hilos", d.threads)?,
        ..d
    };
    if let Some(s) = args.take("seleccion") {
        cfg.selection = parse_selection(&s)?;
    }
    if let Some(s) = args.take("cruce") {
        cfg.crossover = parse_crossover(&s)?;
    }
    if let Some(s) = args.take("agregacion") {
        cfg.fitness.aggregation = parse_aggregation(&s)?;
    }
    if let Some(s) = args.take("control") {
        cfg.operator_control = match s.as_str() {
            "uniforme" => OperatorControl::Uniform,
            "bandido" => OperatorControl::Bandit,
            _ => return Err(Error::Config(format!("control desconocido: «{s}»"))),
        };
    }
    if let Some(s) = args.take("nicho") {
        cfg.niching = match s.split_once(':') {
            None if s == "ninguno" => crate::ga::Niching::None,
            None if s == "crowding" => crate::ga::Niching::Crowding,
            None if s == "crowding-seleccion" => crate::ga::Niching::CrowdingSelective,
            None if s == "sharing" => crate::ga::Niching::Sharing(0.15),
            Some(("sharing", v)) => crate::ga::Niching::Sharing(num(v, "sharing")?),
            _ => return Err(Error::Config(format!("nicho desconocido: «{s}»"))),
        };
    }
    if let Some(s) = args.take("estilo") {
        cfg.fixed_style = Some(
            crate::level::Style::from_name(&s)
                .ok_or_else(|| Error::Config(format!("estilo desconocido: «{s}»")))?,
        );
    }
    let t = &mut cfg.fitness.targets;
    t.difficulty = args.get("dificultad", t.difficulty)?;
    if cfg.population < 2 || cfg.elitism >= cfg.population {
        return Err(Error::Config(
            "se necesita población ≥ 2 y elitismo < población".into(),
        ));
    }
    if cfg.width < 2 * crate::level::SAFE_COLS + 8 {
        return Err(Error::Config("el ancho mínimo es 14 columnas".into()));
    }
    Ok(cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Args {
        Args::parse(s.split_whitespace().map(String::from)).expect("válidos")
    }

    #[test]
    fn opciones_y_posicionales() {
        let mut a = args("evolucionar x.txt --semilla 5 --reparar no --hilos=2");
        assert_eq!(a.command, "evolucionar");
        assert_eq!(a.positional, vec!["x.txt"]);
        assert_eq!(a.get("semilla", 0_u64).ok(), Some(5));
        assert_eq!(a.flag("reparar", true).ok(), Some(false));
        assert_eq!(a.get("hilos", 0_usize).ok(), Some(2));
        assert!(a.finish().is_ok());
        let a = args("x --desconocida 1");
        assert!(a.finish().is_err());
    }

    #[test]
    fn selecciones() {
        assert_eq!(
            parse_selection("torneo:5").ok(),
            Some(Selection::Tournament(5))
        );
        assert_eq!(
            parse_selection("ruleta:lineal:2:sus").ok(),
            Some(Selection::Roulette {
                scaling: Scaling::Linear(2.0),
                sus: true
            })
        );
        assert_eq!(
            parse_selection("ruleta").ok(),
            Some(Selection::Roulette {
                scaling: Scaling::None,
                sus: false
            })
        );
        assert!(parse_selection("ruleta:rara").is_err());
    }

    #[test]
    fn configuracion_completa() {
        let mut a = args(
            "e --poblacion 50 --seleccion ruleta:sigma --cruce bloque --agregacion deb --control bandido",
        );
        let c = ga_config(&mut a).expect("válida");
        assert_eq!(c.population, 50);
        assert_eq!(c.crossover, Crossover::Block);
        assert_eq!(c.fitness.aggregation, Aggregation::Deb);
        assert!(a.finish().is_ok());
    }
}
