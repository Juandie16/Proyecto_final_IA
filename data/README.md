# Datos

**Fuente:** *Accidentes de Tránsito Ocurridos en la Ciudad de Cartagena de Indias* (2020–2025).
**Publicador:** Departamento Administrativo de Tránsito y Transporte (DATT), Alcaldía de Cartagena.
**Portal:** https://datosabiertos.cartagena.gov.co/dataset/accidentes-de-transito-ocurridos-en-el-municipio-de-cartagena-de-indias
(también listado en datos.gov.co, id `5z62-bn99`).
**Licencia:** Creative Commons Attribution (CC-BY): se puede usar citando la fuente.
**Tamaño informado por el portal:** 1,1 MiB, 25 columnas, formato XLSX; actualización trimestral (última: 5 de octubre de 2026).
**Datos personales:** el diccionario no incluye nombres ni documentos de identidad; incluye ubicación (barrio, vía, coordenadas) del siniestro.

**Contenido real verificado al ejecutar el pipeline (2026-11-16):** 8.066 filas; tras eliminar 573 IPAT duplicados quedan 7.493 siniestros. La columna `gravedad` solo toma los valores «Con heridos» (6.924) y «Con muertos» (569) tras deduplicar; **no hay siniestros de solo daños materiales**, por lo que el proyecto estima si un siniestro con víctimas será fatal.

**Hallazgo de calidad (verificado con `python -m src.data_quality`):** de 2020 a 2024 la columna `muertos` es coherente con `gravedad` (0 incoherencias). En 2025 no lo es: 282 de 1.390 filas (264 «Con heridos» con `muertos` > 0 y 18 «Con muertos» con `muertos` = 0), y la suma de `muertos` ese año (529) es incompatible con las cifras oficiales del DATT. Se descartó que `muertos` sea una copia de `heridos` (0 % de coincidencia). Se usa `gravedad` como fuente de la etiqueta y `muertos` solo se audita. Pendiente: consultar al DATT.

Los archivos se descargan con `python -m src.data_download`, que además registra URL, fecha y hash SHA-256 en `data/raw/SOURCE.json`.
`data/raw/` y `data/processed/` no se versionan: se regeneran con los scripts.

## Diccionario resumido (según el publicador)
`ipat` (id) · `fecha` · `hora` · `gravedad` · `clase_accidente` · `choque_con` ·
`objeto_fijo` · `hipotesis_1…4` · `BARRIO` · `lugar` · `via1…via6` (calle, carrera, diagonal, transversal, avenida, corredor de carga) ·
`localidad` · `muertos` · `heridos` · `coordenadas` · `latitud` · `longitud`.

Cita sugerida: Alcaldía de Cartagena de Indias, DATT (2026). *Accidentes de tránsito ocurridos en la ciudad de Cartagena de Indias, 2020–2025* [Conjunto de datos]. Portal de Datos Abiertos.

## Codificaciones del archivo que el preprocesamiento corrige (verificado 2026-11-22)
- `via1…via6`: el vacío viene como `"0"`. En 2020–2022 `via1…via5` y `lugar` están vacíos en el 100 % de las filas, por lo que
  `tipo_via` es «desconocido» en esos años. `via6` no es un nombre de vía: vale «Corredor de Carga», «Otro» o `"0"`.
- `choque_con` y `objeto_fijo`: «sin dato» se escribe distinto según el año (`"0"` en 2020–22, vacío en 2023–24, «No» en 2025;
  8 celdas de 2025 contienen el nombre de la columna, «Choque con»). Se unifican en `sin_dato`. `objeto_fijo` queda con un 94 % de `sin_dato`.
- `localidad` viene como código 1, 2, 3. Su traducción a nombre (solo para reportes) está en `src/config.py`
  (`LOCALIDAD_NOMBRES`), marcada [CONFIRMAR]: el DATT no publica el diccionario de este campo.
- 26 filas tienen coordenadas fuera de la caja de Cartagena o ausentes (p. ej. longitud `-754751`, latitud `0`, o puntos de Barú y de la
  zona rural norte que quedan fuera de la caja). Se dejan como faltantes (se imputa la mediana).
