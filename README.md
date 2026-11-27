# GA Levels Generator (`nivelga`)

Generación automática de niveles de plataformas 2D con un algoritmo genético. El
cromosoma es la matriz de tipos lógicos de tiles más un gen que elige el estilo
visual entre los cuatro packs *Pixel Platformer* de Kenney. La función de
adaptación busca niveles jugables con dificultad moderada, variedad estructural
y coherencia visual.

Todo está escrito a mano en Rust, sin dependencias externas: generador
aleatorio, lectura y escritura de PNG (incluido DEFLATE), verificador de
jugabilidad, métricas, operadores genéticos, redes neuronales, estadística y
gráficas.

## Requisitos

- Rust 1.100 o posterior (edición 2024).

## Uso

```sh
cargo run --release -- ayuda
cargo run --release -- evolucionar --semilla 3 --salida resultados/prueba
cargo run --release -- ver resultados/prueba/mejor.txt
cargo run --release -- dibujar resultados/prueba/mejor.txt mejor.png --escala 2
cargo run --release -- comprobar resultados/prueba/mejor.txt
```

`evolucionar` guarda en el directorio de salida el historial por generación
(`historial.csv`), las gráficas (`aptitud.svg`, `objetivos.svg`,
`diversidad.svg` y `estilos.svg`) junto con el mejor nivel en texto y en PNG
con su trayectoria óptima. También guarda el mejor nivel jugable de cada estilo.

## Jugar los niveles (Godot 4)

El juego en `godot/` es un plataformas sencillo con la misma física que el
verificador (salto de unas 3 casillas de alto y 4 de largo). Pide a `nivelga
generar` un nivel nuevo y distinto cada vez (ancho, dificultad objetivo y
tileset al azar) y prepara el siguiente en segundo plano mientras juegas, así
hay niveles sin fin. Cada partida se registra en `resultados/partidas.csv`.

```sh
cargo build --release
sh scripts/jugar.sh                     # niveles nuevos sin fin
sh scripts/jugar.sh --nivel=RUTA.json   # un nivel concreto (ver `nivelga exportar`)
cargo run --release -- partidas         # compara la dificultad D con tus partidas
```

Controles: flechas o A/D para moverse, espacio, Z o flecha arriba para saltar,
flechas arriba y abajo en las escaleras, R reinicia, N o Enter pasa al
siguiente nivel, F marca el nivel para revisarlo (se copia en
`resultados/marcados/`), F3 o Tab activa el modo depuración y Esc sale. Con la
flecha abajo se baja de una plataforma de un solo sentido.

El modo depuración dibuja el contorno de cada celda según su tipo (suelo,
plataforma de un sentido, escalera, peligro, moneda, meta y decoración sin
colisión), la caja de choque de cada enemigo con su zona de patrulla y la caja
del jugador.

Godot solo se usa para jugar: la generación, las métricas y el aprendizaje
siguen hechos a mano en Rust. El script busca el ejecutable en la variable
`GODOT`, en el `PATH` o en la carpeta de la versión anterior del proyecto.

## Red de naturalidad (cross-entropy)

Un MLP aprende a predecir cada tile a partir de su contexto en los niveles de
Super Mario Bros del VGLC. Su log-verosimilitud negativa se usa como término de
aptitud para que los niveles se parezcan a niveles diseñados por personas.

```sh
sh scripts/descargar_vglc.sh            # datos (no se versionan)
cargo run --release -- entrenar-nn      # guarda modelos/naturalidad.nng
cargo run --release -- evolucionar --naturalidad modelos/naturalidad.nng
```

## Agente de aprendizaje por refuerzo (A2C)

Un agente actor-crítico aprende a jugar los niveles con las mismas acciones que
el verificador (caminar, escaleras y los 27 arcos de salto). Su pérdida de
política es la cross-entropy de la acción tomada ponderada por la ventaja. Se
entrena con niveles nuevos en cada episodio (ventanas mutadas de niveles del
GA y niveles aleatorios) y un currículo de longitud creciente. Sirve como
probador: su tasa de victorias mide la dificultad empírica de un nivel.

```sh
cargo run --release -- entrenar-rl --pasos 20000000 --ejecuciones-ga 64
cargo run --release -- jugar resultados/prueba/mejor.txt
cargo run --release -- evolucionar --agente modelos/agente.rla --objetivo-victorias 0.6
```

## Programación Genética: clasificador simbólico de jugabilidad

La PG evoluciona una expresión sobre 21 rasgos estructurales que predice si un
nivel es jugable sin ejecutar el BFS. Su aptitud es la cross-entropy binaria
con escalado lineal (Keijzer, 2003) más una presión de parsimonia. Se compara
con una regresión logística y un MLP entrenados con la misma pérdida.

```sh
cargo run --release -- entrenar-pg      # informe en resultados/pg
```

## Experimentos

```sh
cargo run --release -- experimentos     # lista e1 a e10
cargo run --release -- experimento e1 --repeticiones 30
cargo run --release -- experimento todos --repeticiones 30 --naturalidad modelos/naturalidad.nng
```

Los resultados obtenidos y su discusión están en [RESULTADOS.md](RESULTADOS.md).

Cada experimento usa las mismas semillas para todas las variantes y guarda el
CSV por ejecución, el resumen, las curvas de convergencia, los diagramas de
caja y un informe con Kruskal-Wallis, Mann-Whitney con Holm, Â12, Fisher y
Friedman.

## Diseño de niveles

El generador no coloca tiles sueltos: usa una biblioteca de patrones de diseño
(`src/patterns.rs`) extraída de la literatura (patrones de Super Mario Bros de
Dahlskog y Togelius, grupos de ritmo de Smith y Whitehead, plantillas de
Spelunky, estructura en cuatro actos de Nintendo) y del análisis de niveles
hechos a mano en proyectos de Godot. Hay huecos con arco de monedas,
multihuecos, escalones, mesetas, zanjas con enemigos, hordas, pinchos, puentes,
cadenas de plataformas, rutas dobles de riesgo y recompensa, pilares y
escaleras de mano para desniveles mayores que el salto. Las plataformas son de
un solo sentido (se atraviesan desde abajo).

El cromosoma sigue siendo la matriz de tiles: los patrones se usan en la
inicialización (grupos de ritmo con descansos), en una mutación que inserta
patrones y en la aptitud, que detecta los patrones de la matriz y mide la
variedad de diseño, la mezcla de tipos, la verticalidad y el ritmo. El cruce
solo corta en columnas de descanso comunes a ambos padres.

## Componentes

| Módulo | Contenido |
| --- | --- |
| `level` | Tiles lógicos, estilos, nivel guardado por columnas, formato de texto |
| `play` | Verificador de jugabilidad: arcos de salto, Dijkstra con cubetas, trayectoria |
| `metrics` | Rasgos, dificultad, variedad de diseño y coherencia |
| `patterns` | Biblioteca de patrones de diseño y detector |
| `fitness` | Agregación multiobjetivo con la jugabilidad como restricción dura |
| `ga` | Selección (torneo y ruleta con escalado y SUS), cruce, mutación, elitismo, reparación, bandido D-MAB |
| `nn` | Red neuronal con cross-entropy, Adam y chequeo de gradientes |
| `naturalness`, `vglc` | Modelo de naturalidad de tiles y lector del corpus VGLC |
| `baselines`, `experiment` | Búsqueda aleatoria, (1+1)-EA y ejecutor de experimentos |
| `rl` | Entorno, agente actor-crítico y entrenamiento A2C |
| `gp`, `surrogate` | Programación Genética, rasgos estructurales y modelos sustitutos |
| `stats` | Mann-Whitney, Â12, Fisher, Kruskal-Wallis, Friedman y Holm |
| `image`, `render`, `plot`, `ansi` | PNG, dibujo con tilesets, gráficas SVG y visor en terminal |

## Calidad

```sh
cargo fmt --check
cargo clippy --all-targets
cargo test
cargo bench --bench evaluacion
```

El código usa `clippy::pedantic` sin avisos y prohíbe `unsafe`. Los
comentarios y mensajes están en español. El test `tests/estilo.rs` comprueba
que no haya rayas ni comas antes o después de la conjunción «y».

## Créditos

Tilesets *Pixel Platformer* y sus expansiones Farm, Food e Industrial de
[Kenney](https://kenney.nl) (CC0).
