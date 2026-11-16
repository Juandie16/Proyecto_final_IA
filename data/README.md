# Datos

**Fuente:** *Accidentes de Tránsito Ocurridos en la Ciudad de Cartagena de Indias* (2020–2025).
**Publicador:** Departamento Administrativo de Tránsito y Transporte (DATT), Alcaldía de Cartagena.
**Portal:** https://datosabiertos.cartagena.gov.co/dataset/accidentes-de-transito-ocurridos-en-el-municipio-de-cartagena-de-indias
(también listado en datos.gov.co, id `5z62-bn99`).
**Licencia:** Creative Commons Attribution (CC-BY): se puede usar citando la fuente.
**Tamaño informado por el portal:** 1,1 MiB, 25 columnas, formato XLSX; actualización trimestral (última: 5 de octubre de 2026).
**Datos personales:** el diccionario no incluye nombres ni documentos de identidad; incluye ubicación (barrio, vía, coordenadas) del siniestro.

Los archivos se descargan con `python -m src.data_download`, que además registra URL, fecha y hash SHA-256 en `data/raw/SOURCE.json`.
`data/raw/` y `data/processed/` no se versionan: se regeneran con los scripts.

## Diccionario resumido (según el publicador)
`ipat` (id) · `fecha` · `hora` · `gravedad` (Solo daños materiales / Con heridos / Con muertos) · `clase_accidente` · `choque_con` ·
`objeto_fijo` · `hipotesis_1…4` · `BARRIO` · `lugar` · `via1…via6` (calle, carrera, diagonal, transversal, avenida, corredor de carga) ·
`localidad` · `muertos` · `heridos` · `coordenadas` · `latitud` · `longitud`.

Cita sugerida: Alcaldía de Cartagena de Indias, DATT (2026). *Accidentes de tránsito ocurridos en la ciudad de Cartagena de Indias, 2020–2025* [Conjunto de datos]. Portal de Datos Abiertos.
