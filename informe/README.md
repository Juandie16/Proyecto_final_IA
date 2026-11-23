# Informe IEEE (hito 3)

Formato IEEE de conferencia, doble columna (`IEEEtran`), en español. Listo para Overleaf.

## Subir a Overleaf
1. Comprima la carpeta `informe/` completa (`main.tex`, `cifras.tex`, `referencias.bib`, `tablas/`, `figuras/`).
2. En Overleaf: *New Project → Upload Project* y suba el `.zip`.
3. Compilador: **pdfLaTeX** (menú *Menu → Compiler*). Overleaf corre BibTeX solo.
4. Objetivo: 6–8 páginas sin contar referencias. El borrador no se compiló localmente (no hay LaTeX en el equipo
   donde se generó); revise la extensión y la ubicación de figuras y tablas en Overleaf.

## Regla: ningún número escrito a mano
Las cifras del texto son macros `\Cifra...` definidas en `cifras.tex`, y las tablas están en `tablas/*.tex`. Ambos se generan
desde `experiments/` con:
```powershell
python -m src.make_report_tables
```
Si cambia un resultado, vuelva a ejecutar ese comando y suba de nuevo `cifras.tex`, `tablas/` y `figuras/`. Las únicas cifras
escritas a mano son datos externos citados (OMS y El Universal/DATT en la introducción).

## Pendientes marcados en rojo
`[COMPLETAR POR EL EQUIPO]` (afiliaciones, contribuciones, uso de IA, reflexión), `[CONFIRMAR]` (usuario objetivo,
hipótesis de errores, nombres de localidad) y `[VERIFICAR]` (sub-registro con el DATT; en `referencias.bib`: páginas de Kaufman
et al., autores y DOI de Iranitalab y Khattak, autor y fecha de consulta de la nota de El Universal).
