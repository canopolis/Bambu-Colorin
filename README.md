# Bambu Colorín

Port del motor full spectrum de **Snapmaker Orca FullSpectrum v0.9.18** (ratdoux) a **Bambu Studio v02.09.01.70**.

- Fuente del motor: tag `v0.9.18` de canopolis/OrcaSlicer-FullSpectrum-Plus
- Base: bambulab/BambuStudio tag `v02.09.01.70` (fija; cambia solo cuando Arturo lo decida)
- Referencia del port ya hecho: canopolis/Prusa-Colorin
- Objetivo: Bambu P3C (cabezales independientes), filamentos CMYW / CMYWB, envío de G-code por WiFi

## Ventaja
Bambu Studio y OrcaSlicer comparten origen (Orca es un fork de Bambu Studio), así que el port debe ser mucho más directo que el de PrusaSlicer.

## Fases
1. Repo + workflow que compila Bambu Studio original (Windows, .exe portable con número de build)
2. Motor libslic3r (MixedFilament, ImageMap, PrintObjectSlice, ToolOrdering, GCode, WipeTower)
3. Importación OBJ con color + diálogo
4. Vista 3D en color + pincel
5. Segmentación, Local-Z, modulación de perímetros
6. Interfaz, perfiles .ini, nombre/icono/instalador
