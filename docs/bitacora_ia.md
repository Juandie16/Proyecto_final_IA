# Bitácora humano–IA

**Equipo:** [Integrante A], [Integrante B] — *reemplazar por los nombres reales.*
**Regla:** cada fila describe un uso relevante de IA generativa. Las columnas «Qué se aceptó, modificó o descartó» y
«Cómo se verificó» deben reflejar **lo que ustedes realmente hicieron**; las que aparecen como *[CONFIRMAR]* las llena el equipo
después de revisar el material.

| Fecha | Integrante | Herramienta | Para qué se usó | Qué se aceptó, modificó o descartó | Cómo se verificó |
|---|---|---|---|---|---|
| 2026-11-16 | [Integrante A] | Claude | Proponer temas de proyecto | La primera lista repetía los ejemplos del PDF del curso, que el equipo había pedido evitar; se descartó completa y se pidió una nueva. | Comparando cada propuesta con la tabla de ejemplos del PDF. |
| 2026-11-16 | [Integrante A] | Claude | Proponer 6 temas nuevos y recomendar uno | Se aceptó el tema 1 (gravedad de siniestros viales). La propuesta decía que los datos estaban en datos.gov.co (ANSV/Policía) y advertía que Cartagena podía tener pocos registros: ambas afirmaciones eran de memoria y resultaron imprecisas. | Búsqueda web y lectura de las fichas del portal: el dataset real lo publica el DATT en el portal de datos abiertos de Cartagena (2020–2025, 1,1 MiB, CC-BY), con 25 columnas. |
| 2026-11-16 | [Integrante A] | Claude | Localizar el dataset y leer su diccionario de datos | Se aceptó el diccionario tal como lo publica el DATT; se descartó el dataset de Facatativá (152 filas), que apareció en la búsqueda y es muy pequeño. | Lectura directa de la ficha del recurso (diccionario, licencia, tamaño, fecha de actualización). Pendiente: abrir el archivo y contar filas reales. [CONFIRMAR] |
| 2026-11-16 | [Integrante B] | Claude | Escribir el código del pipeline (`src/`), las pruebas y la evaluación predeclarada | [CONFIRMAR qué se modificó tras revisarlo línea por línea]. Decisiones de diseño propuestas por la IA: etiqueta binaria, partición temporal, exclusión de `muertos`/`heridos` por fuga, test bloqueado tras `--final`. | 10 pruebas unitarias con datos **sintéticos** pasan. **No se ha ejecutado con datos reales**; la IA no tuvo acceso de red. El equipo debe correr los 4 comandos del README y revisar la salida. |
| 2026-11-16 | [Integrante B] | Claude | Redactar la página de avance del hito 2 | [CONFIRMAR]. | Los números del PDF se leen de `experiments/*.json`, no se escriben a mano. Los umbrales de éxito son estimaciones iniciales a validar por el equipo. |

## Reflexión (borrador para el informe, media página)
En esta etapa la IA propuso algo inadecuado en dos ocasiones. Primero, su lista inicial de temas reutilizó los ejemplos del PDF
del curso, lo que ignoraba el propósito de proponer un problema propio; lo notamos al compararla con la tabla de ejemplos.
Segundo, al recomendar el tema final afirmó de memoria que los datos provenían de datos.gov.co (ANSV/Policía) y que Cartagena
podría tener pocos registros. Al verificar en el portal encontramos que el DATT publica el dataset de Cartagena con 25 columnas
y licencia CC-BY, así que la afirmación era imprecisa. Lección: tratar como hipótesis cualquier dato que la IA diga «de memoria»
y comprobarlo en la fuente primaria antes de comprometer el diseño. *[Ampliar con un caso propio del equipo cuando ocurra.]*
