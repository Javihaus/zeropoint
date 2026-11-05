# Datos sintéticos de ZeroPoint 1.0: criterio de creación

## Para qué sirven

Los errores del modelo conjunto frente a Laya no son de cantidad de datos sino de tres huecos concretos, vistos caso a caso en los tests:

1. **Etiquetas de categoría amplia.** Cuando las opciones son departamentos o categorías ("Human Resources", "search creative work"), el modelo acierta poco y casi siempre contesta "ninguna" con mucha seguridad. Nuestros datos de entrenamiento solo tienen intenciones finas ("activate card").
2. **Textos largos tipo correo**, con asunto y cuerpo. Todo nuestro entrenamiento son frases cortas.
3. **Negaciones y contrastes** ("reserva el hotel pero no el coche"). El modelo se va a la opción que se niega.

Los 5.000 ejemplos cubren exactamente esos tres huecos, en inglés y en castellano. No pretenden enseñar nada más.

## Reparto

| Familia | Idioma | Ejemplos | Lotes (número global) |
|---|---|---|---|
| Tickets a departamentos | inglés | 1.500 | 1–15 |
| Tickets a departamentos | castellano | 700 | 16–22 |
| Negaciones y contrastes | inglés | 1.100 | 23–33 |
| Negaciones y contrastes | castellano | 500 | 34–38 |
| Categorías amplias con nombres propios | inglés | 900 | 39–47 |
| Categorías amplias con nombres propios | castellano | 300 | 48–50 |

El castellano lleva un 30 % del total porque es donde el modelo cae más en listas largas.

## Formato de cada ejemplo

Una línea JSON por ejemplo:

```json
{"id": "b07-042", "batch": 7, "family": "ticket", "lang": "en", "pool": "tk_saas",
 "text": "Subject: ...\n...", "label": "Bug Reports", "hard_negatives": ["API and Integrations"],
 "scores": {"realism": 4, "label": 5, "difficulty": 3, "diversity": 4, "language": 5, "form": 4}}
```

- **pool**: el conjunto de etiquetas al que pertenece (fichero `pools.json`). En el entrenamiento, las demás opciones de la lista salen del mismo conjunto, así que las opciones son siempre del mismo mundo: un ticket de banco compite contra departamentos de banco.
- **label**: la respuesta correcta. Solo puede ser `null` en la familia de negaciones, cuando el texto niega todas las intenciones posibles (máximo 15 %).
- **hard_negatives**: de 1 a 3 etiquetas del mismo conjunto con las que es más fácil confundirse. El entrenamiento las mete siempre en la lista. En la familia de negaciones, la intención negada está siempre aquí.
- **Casos de "ninguna"**: no se escriben aparte. El entrenamiento los fabrica quitando la etiqueta correcta de la lista y dejando las confusiones. Así el modelo aprende que "reserva el hotel pero no el coche", con solo "alquilar un coche" en la lista, es "ninguna".

## Conjuntos de etiquetas

Son 30 conjuntos (`pools.json`), de 6 a 12 etiquetas cada uno, inventados para este trabajo:

- **Tickets en inglés (8):** software, empresa interna, tienda online, banco, telefonía, suministros, universidad y clínica.
- **Tickets en castellano (5):** telefonía, ayuntamiento, empresa interna, tienda online y seguros.
- **Negaciones (6 en inglés, 4 en castellano):** viajes, banca, hogar, compras, trabajo y comida.
- **Categorías (5 en inglés, 2 en castellano):** ocio, lugares, hogar, gestiones personales, preguntas de información y vida diaria.

**Ninguna etiqueta coincide con una etiqueta de los tests.** Los nombres de cola de los tests de tickets ("Technical Support", "Human Resources", "Billing and Payments"…) están prohibidos, y lo comprueba un programa. Los departamentos tienen nombres distintos en cada conjunto ("People Team", "Personas y Cultura", "Payroll", "Nóminas"), para que el modelo aprenda qué hace un departamento y no a reconocer un nombre concreto.

## Reglas de escritura por familia

### Tickets
- **Forma:** empiezan por `Subject:` o `Asunto:` y siguen con el cuerpo en otra línea. Tienen entre 25 y 150 palabras.
- **Lo que necesita el cliente se deduce por la situación, sin nombrar el departamento:** un monitor que no detecta la segunda pantalla va a soporte informático aunque la palabra no aparezca. Como máximo un 10 % de los textos puede contener su etiqueta literal.
- **Variedad de redacción:** educados, secos, enfadados, con faltas leves, con datos inventados (números de pedido, importes, fechas). Unos ponen lo importante en el asunto y otros en el cuerpo.
- **Casos frontera a propósito.** Al menos un 25 % podría parecer de un departamento vecino, y el detalle decide. Por ejemplo, un cobro duplicado de un producto defectuoso: si piden el dinero va a reembolsos, y si piden otro producto va a producto defectuoso.
- **Datos personales:** ninguno real. Nombres, teléfonos y cuentas son inventados y verosímiles.

### Negaciones y contrastes
- **Siempre aparecen al menos dos intenciones del conjunto**, una pedida y otra negada o descartada ("no quiero cancelar, solo cambiar la fecha").
- **Formas de negar variadas:** "no", "sin", "en vez de", "ya no hace falta", "olvida lo de…", "no es que…, es que…", orden invertido (primero lo negado), doble mención, ironía suave.
- **La intención negada va siempre en `hard_negatives`.**
- **Hasta un 15 % niega todo** ("no necesito cambiar el PIN, ya lo hice ayer") y lleva `label: null`.
- **Longitud:** de 4 a 45 palabras, como lo que alguien escribe o dicta a un asistente.

### Categorías amplias
- **El texto gira alrededor de un nombre propio o un dato concreto:** un título de película o libro, un grupo de música, un equipo, un restaurante, una ciudad.
- **Hay que saber qué es ese nombre para elegir la categoría.** "Busca Cien años de soledad" va a libros, y "pon Hotel California" va a música, no a alojamiento.
- **Al menos un 25 % son trampas a propósito:** nombres que parecen de otra categoría ("Hotel California", "The Kitchen" como serie, un libro llamado como una ciudad).
- **Los títulos y nombres son reales y conocidos** cuando hace falta conocimiento del mundo. Solo se usan como referencia, nunca se copia texto de las obras.
- **Longitud:** de 4 a 45 palabras.

## Rúbrica de puntuación

Cada ejemplo recibe una nota de 1 a 5 en seis dimensiones:

| Dimensión | 5 significa | 1 significa |
|---|---|---|
| **realism** | lo podría haber escrito una persona real en esa situación | suena a ejercicio o a plantilla |
| **label** | la etiqueta es la única correcta del conjunto y un experto no dudaría | la etiqueta es discutible o hay otra igual de buena |
| **difficulty** | hay que entender la situación para acertar; no basta con buscar palabras | el texto repite la etiqueta o una palabra clave que la delata |
| **diversity** | no se parece a ningún otro del lote en estructura, tema ni arranque | es casi una copia de otro ejemplo |
| **language** | idioma natural y correcto, registro coherente (en castellano, uso de España) | traducción literal, giros raros, errores no buscados |
| **form** | cumple las reglas de la familia (longitud, asunto, negación presente, nombre propio) | incumple una regla de la familia |

La nota la pone quien escribe el lote, releyendo cada ejemplo después de escribir los 100. Se pide ser estricto: en un lote normal deberían aparecer notas de 2 y 3.

## Proceso iterativo

Cada lote son 100 ejemplos. El modo depende del número global de lote:

- **Lotes impares, el peor:**
  1. Se escriben y puntúan los 100.
  2. El programa `tools.py select` elige el ejemplo de suma más baja. Los empates se resuelven por la nota de etiqueta y luego por el identificador.
  3. Elige también los 10 ejemplos más parecidos a él en el espacio de las seis notas, es decir, los que fallan de la misma manera.
  4. Se escribe por qué el peor es el peor. Se reescriben los 11 corrigiendo esa causa y se vuelven a puntuar.
- **Lotes pares, el mejor:**
  1. Se escriben y puntúan los 100.
  2. El programa elige el ejemplo de suma más alta y los 10 más alejados de él en el espacio de notas.
  3. Se escribe qué hace bueno al mejor. Los 10 se reescriben tomándolo como referencia de calidad, pero manteniendo su tema, para no perder variedad, y se vuelven a puntuar.

**Por qué "parecido" se mide con las notas y no con el texto.** Lo que se quiere corregir es una forma de fallar, no un tema. Dos ejemplos de temas distintos con la misma etiqueta discutible fallan igual.

**Registro de cada lote** (`log/bNN.json`):
- el modo;
- el ejemplo ancla con su texto y notas;
- la causa escrita;
- para cada ejemplo reescrito, el texto antes y después, con las notas antes y después.

Cada lote debe pasar `tools.py check` sin errores antes y después de las reescrituras.

## Controles de fuga hacia los tests

1. **Quien escribe no ve los tests.** Las instrucciones de cada lote no incluyen ni un ejemplo de test.
2. **Etiquetas:** ninguna coincide con una etiqueta de los tests. Lo comprueba el programa.
3. **Textos:** al final se calcula, para cada ejemplo sintético, la frase más parecida de todos los tests, con el codificador e5. Se reescribe cualquier ejemplo con similitud de 0,92 o más.
4. **Duplicados:** se eliminan los textos repetidos dentro de todo el conjunto, tras normalizar mayúsculas, acentos y signos.

## Licencia

Texto escrito para ZeroPoint, sin material de terceros. Se publica con la misma licencia que el modelo. Los títulos de obras y los nombres de lugares solo aparecen como referencia.

## Resultado

**Volumen:** 5.000 ejemplos en 50 lotes de 100, 25 en modo "peor" y 25 en modo "mejor".

| Familia | Inglés | Castellano | Palabras (mínimo / mediana / máximo) |
|---|---|---|---|
| Tickets | 1.500 | 700 | 25 / 44 / 134 |
| Negaciones | 1.100 | 500 | 7 / 17 / 34 |
| Categorías | 900 | 300 | 4 / 9 / 23 |

- **Negaciones que lo niegan todo** (`label: null`): 138, un 8,6 % de su familia.
- **Textos que contienen su etiqueta literal:** 27, un 0,5 %.
- **Cada etiqueta de cada conjunto aparece al menos 9 veces.** Los recuentos por etiqueta están en `stats.json`.

**Reescrituras del proceso iterativo:** 525 ejemplos.
- La suma de notas sube de media 3,2 puntos por ejemplo reescrito, sobre 30.
- En 6 casos no subió.
- Cada lote tiene su registro en `log/bNN.json`, con el ancla, la causa y el texto antes y después.

**Reparto final de notas** (número de ejemplos con 1 / 2 / 3 / 4 / 5):

| Dimensión | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| realism | 0 | 1 | 397 | 3.323 | 1.279 |
| label | 0 | 0 | 71 | 2.263 | 2.666 |
| difficulty | 0 | 43 | 575 | 3.691 | 691 |
| diversity | 0 | 0 | 213 | 3.510 | 1.277 |
| language | 0 | 0 | 33 | 1.645 | 3.322 |
| form | 0 | 0 | 0 | 199 | 4.801 |

**Las notas son benévolas.** Las pone quien escribe, y las notas de 2 casi no aparecen, aunque se pedían. En los lotes en modo "peor", el ancla es el ejemplo más flojo del lote, pero no siempre un ejemplo malo. Las notas sirven para ordenar dentro de un lote, no como medida absoluta de calidad. La medida que cuenta es el resultado del modelo en los tests.

**Fuga hacia los tests:**
- La primera pasada encontró 24 ejemplos con similitud de 0,92 o más con alguna frase de los tests.
- Ninguno era una copia: eran peticiones del mismo tipo y con palabras parecidas ("cancela la mesa de esta noche", "pon una alarma para las seis").
- Se reescribieron con otra situación y otras palabras (`log/leakage_rewrites.json`).
- Tras reescribirlos, la similitud máxima es 0,919 y la mediana 0,871. Con este codificador, dos frases cualesquiera del mismo tema dan alrededor de 0,85.

**Fichero de entrenamiento:** `synth.jsonl`, una línea por ejemplo con `id`, `family`, `lang`, `pool`, `text`, `label` y `hard_negatives`.
