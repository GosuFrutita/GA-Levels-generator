# Resultados experimentales

Todas las cifras se obtienen con `cargo run --release -- experimento todos
--repeticiones 30 --naturalidad modelos/naturalidad.nng`. Cada variante se
ejecuta con las mismas 30 semillas (números aleatorios comunes) y un
presupuesto fijo de 20 000 evaluaciones (población 100, 200 generaciones,
2 élites). Se informa la mediana con su rango intercuartílico. Las variantes se
comparan con Kruskal-Wallis, con Mann-Whitney ajustado por Holm frente a la
referencia y con el tamaño de efecto Â12 de Vargha y Delaney (0,5 = sin efecto;
umbrales 0,56, 0,64 y 0,71 para efecto pequeño, mediano y grande). Los informes
completos, los CSV y las gráficas quedan en `resultados/experimentos/eN/`.

## Algoritmo genético

| Exp. | Pregunta | Resultado |
|---|---|---|
| E1 | ¿Aporta el GA? | El GA (0,9943) supera a la búsqueda aleatoria (0,9619; Â12 = 1,00). Frente al GA sin cruce (0,9931) la diferencia no es significativa (p = 0,22). El (1+1)-EA queda ligeramente por encima (0,9961; Â12 = 0,25 a su favor). |
| E2 | Ruleta frente a torneo | El torneo de 2 gana a todas las ruletas (Â12 ≥ 0,87). Torneo de 5 o 7 (0,996) > torneo de 3 (0,994) > torneo de 2 (0,986) > ruletas (0,977 a 0,979). |
| E3 | Elitismo | Sin élites la aptitud cae (0,9735; Â12 = 1,00 frente a 2 élites). Más élites mejoran algo (10 élites: 0,9953; Â12 = 0,28 a su favor). |
| E4 | Cruce | Columnas (0,9943) y bloque 2D (0,9951) empatan (p = 0,19). El cruce uniforme es peor (0,9841; Â12 = 0,93). |
| E5 | Sensibilidad | La probabilidad de cruce entre 0,6 y 1,0 apenas influye. Con 5 mutaciones por hijo la aptitud baja (0,984 a 0,987 frente a 0,995 a 0,996). |
| E6 | Agregación | Suma ponderada, media geométrica, penalización y reglas de Deb no difieren (Kruskal-Wallis p = 0,34). |
| E7 | Nueve instancias (D* × ancho) | Friedman p = 9,8·10⁻⁶. Rangos medios: (1+1)-EA 3,9; torneo-3 3,1; ruleta 2,0; aleatoria 1,0. |
| E9 | Rango expresivo | La dificultad del mejor nivel cae entre 0,496 y 0,510 con objetivo 0,5. Estilos elegidos: pradera 18, granja 10, dulce 2. |
| E10 | Bandido D-MAB | Sin diferencia frente al operador uniforme (p = 0,25). |
| E11 | Diversidad | Fitness sharing multiplica la diversidad final por 9 a 11 (0,19 a 0,23 frente a 0,021) con menor aptitud (0,971 a 0,975). El crowding con torneo alcanza la mejor aptitud (0,9953; Â12 = 0,27 a su favor). |

**Intensidad de selección (E2).** La medida al seleccionar es de 0,45 a 0,55 en
los torneos y de 0,43 a 0,47 en las ruletas. Ordena bien los métodos pero queda
por debajo de la teórica de Blickle y Thiele (0,58 con t = 2 y 1,34 con t = 7)
y apenas crece con t. La fórmula supone una aptitud normal y continua. Aquí la
población converge pronto, acumula clones con la misma aptitud (elitismo) y se
concentra cerca de 1. Con empates un torneo grande no distingue más que uno
pequeño.

## Componentes de aprendizaje

**Red de naturalidad (cross-entropy).** MLP de 7 846 parámetros entrenado
sobre 12 niveles de Super Mario Bros del VGLC y validado con otros 3. NLL de
validación por tile: 0,129, frente a 0,139 de una cadena de Markov 2D y 0,467
del unigrama. En E8 el término de naturalidad sube la naturalidad del mejor
nivel de 0,529 a 0,815 (Â12 = 1,00) sin perder jugabilidad, dificultad ni
coherencia.

**Agente A2C (cross-entropy ponderada por la ventaja).** 20 millones de pasos
en unos 5 minutos de CPU con niveles nuevos en cada episodio y la misma física
que el verificador (incluidas las zonas de patrulla de los enemigos). Gana el
98,3 % de los niveles aleatorios nuevos jugables (0,2 % sin entrenar) y el
24,4 % de los niveles nuevos evolucionados por el GA, que están optimizados
hacia la dificultad objetivo con muchos saltos ajustados encadenados. Con una
tasa de victorias tan alta su correlación con la dificultad D desaparece
(Spearman −0,03 en aleatorios y −0,10 en evolucionados): como probador sirve
para confirmar que los niveles se pueden pasar, no para ordenarlos por
dificultad. Nunca gana un nivel que el verificador declara injugable, lo que
confirma que ambos usan la misma física.

**Programación Genética.** Clasificador simbólico de jugabilidad entrenado con
cross-entropy binaria, escalado lineal de Keijzer y presión de parsimonia,
sobre 20 000 niveles equilibrados etiquetados por el verificador. En prueba:

| Modelo | AUC | Exactitud | Tamaño |
|---|---|---|---|
| Regresión logística | 0,918 | 0,844 | 22 parámetros |
| MLP | 0,926 | 0,848 | 737 parámetros |
| PG | 0,913 | 0,840 | 15 nodos |

La expresión de la PG es legible: un nivel es menos jugable cuanto mayor es su
vano sin apoyo, cuantos más enemigos tiene (sobre todo junto a los huecos) y
cuanto más alta es su mayor subida, y lo es más cuanto más largo es su mayor
tramo llano.

## Hallazgos sobre el diseño de niveles

- Las primeras versiones de la aptitud premiaban el ruido. Una penalización de
  coherencia con tope formaba una meseta (F = 0,8785 en las 30 semillas) y un
  objetivo de variedad fijado a mano pedía seis veces más patrones distintos
  que un nivel humano (mediana 0,056 en el VGLC). Calibrar con niveles humanos
  y medir la variedad con patrones de diseño resolvió ambos problemas.
- El gen de tileset convergía casi siempre a pradera (27 de 30) porque los
  niveles iniciales encajaban mejor con su personalidad. Con el generador de
  patrones el reparto es más equilibrado (18, 10 y 2), aunque industrial sigue
  sin elegirse.
- El verificador trataba a los enemigos como estáticos, pero en el juego
  patrullan una casilla a cada lado. Una fila de enemigos a 3 columnas parecía
  saltable y en el juego cerraba el paso. Ahora la zona de patrulla es mortal
  para el verificador y el agente, y los enemigos de una fila quedan a 5
  columnas como mínimo.
- Una moneda alcanzable no es necesariamente recogible: puede estar en un pozo
  sin salida. Un recorrido hacia atrás desde la meta distingue las monedas
  seguras y mide los callejones, que ahora penalizan la coherencia.

## Limitaciones

- **Efecto techo.** Con la inicialización por patrones casi todas las variantes
  llegan a F ≈ 0,99, de modo que varias comparaciones discriminan poco. Un
  objetivo más exigente (por ejemplo la naturalidad o el probador de RL como
  términos de la aptitud) daría más margen.
- **El probador de RL** gana casi todos los niveles aleatorios pero solo uno de
  cada cuatro evolucionados, y su tasa de victorias ya no ordena los niveles
  por dificultad; haría falta un currículo con más niveles evolucionados.
- **Las cifras de diseño** (envolvente de salto, mezcla objetivo de patrones,
  rangos de cada estilo) son propuestas razonadas a partir de las fuentes y del
  corpus, no valores publicados.

## Fuentes

Las notas de investigación con todas las referencias están en
`research_notes/` (fuera del repositorio): diseño de niveles de plataformas,
análisis de proyectos de Godot, PCG evolutivo, operadores y evaluación de GA,
redes con cross-entropy, aprendizaje por refuerzo y convenciones de Rust.
