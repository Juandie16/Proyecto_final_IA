# Datos

**Fuente:** *Accidentes de Tránsito Ocurridos en la Ciudad de Cartagena de Indias* (2020–2025).
**Publicador:** Departamento Administrativo de Tránsito y Transporte (DATT), Alcaldía de Cartagena.
**Portal:** https://datosabiertos.cartagena.gov.co/dataset/accidentes-de-transito-ocurridos-en-el-municipio-de-cartagena-de-indias
(también listado en datos.gov.co, id `5z62-bn99`).
**Licencia:** Creative Commons Attribution (CC-BY): se puede usar citando la fuente.
**Tamaño informado por el portal:** 1,1 MiB, 25 columnas, formato XLSX; actualización trimestral (última: 5 de octubre de 2026).
**Datos personales:** el diccionario no incluye nombres ni documentos de identidad; incluye ubicación (barrio, vía, coordenadas) del siniestro.

**Contenido real verificado al ejecutar el pipeline (2026-11-16):** 8.066 filas; tras eliminar 573 IPAT duplicados quedan 7.493 siniestros. La columna `gravedad` solo toma los valores «Con heridos» (6.925) y «Con muertos» (568); **no hay siniestros de solo daños materiales**, por lo que el proyecto estima si un siniestro con víctimas será fatal.

**Hallazgo de calidad (verificado con `python -m src.data_quality`):** de 2020 a 2024 la columna `muertos` es coherente con `gravedad` (0 incoherencias). En 2025 no lo es: 282 de 1.390 filas (264 «Con heridos» con `muertos` > 0 y 18 «Con muertos» con `muertos` = 0), y la suma de `muertos` ese año (529) es incompatible con las cifras oficiales del DATT. Se descartó que `muertos` sea una copia de `heridos` (0 % de coincidencia). Se usa `gravedad` como fuente de la etiqueta y `muertos` solo se audita. Pendiente: consultar al DATT.

Los archivos se descargan con `python -m src.data_download`, que además registra URL, fecha y hash SHA-256 en `data/raw/SOURCE.json`.
`data/raw/` y `data/processed/` no se versionan: se regeneran con los scripts.

## Diccionario resumido (según el publicador)
`ipat` (id) · `fecha` · `hora` · `gravedad` · `clase_accidente` · `choque_con` ·
`objeto_fijo` · `hipotesis_1…4` · `BARRIO` · `lugar` · `via1…via6` (calle, carrera, diagonal, transversal, avenida, corredor de carga) ·
`localidad` · `muertos` · `heridos` · `coordenadas` · `latitud` · `longitud`.

Cita sugerida: Alcaldía de Cartagena de Indias, DATT (2026). *Accidentes de tránsito ocurridos en la ciudad de Cartagena de Indias, 2020–2025* [Conjunto de datos]. Portal de Datos Abiertos.
