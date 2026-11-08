# ADR-003. Almacenamiento y despliegue: datos y modelo versionados, ejecución local por lotes

- **Estado:** aceptada
- **Fecha:** 8 de noviembre de 2026

## Contexto

El dataset es pequeño (5,1 MB) y de licencia abierta (CC BY 4.0). El modelo no cumple los criterios
de equidad (AC4 y AC5), así que no debe tomar decisiones automáticas sobre personas reales.

## Decisión

- **Datos:** el CSV original se versiona en `data/raw/adult.csv`, con su fuente, licencia y suma
  SHA-256 en `data/README.md`. No se modifica: toda transformación ocurre en código.
- **Modelo:** se serializa con `joblib` en `models/modelo_logistico.joblib`, junto con el umbral y la
  lista de variables, para que la predicción sea idéntica a la evaluada.
- **Resultados:** métricas y auditoría en `reports/` (JSON y CSV) y figuras en `reports/figures/`.
- **Despliegue:** ejecución local por lotes (`python -m src.run_all`). No se publica una API.

## Alternativas consideradas

| Alternativa | Por qué no |
|---|---|
| Descargar los datos en cada ejecución | La disponibilidad del repositorio de UCI no está garantizada y el resultado dejaría de ser reproducible |
| DVC o almacenamiento en la nube | Complejidad innecesaria para 5 MB |
| Publicar ya una API (por ejemplo, FastAPI) | Prematuro: el modelo no supera los criterios éticos y una API facilitaría usarlo sin revisión humana |

## Consecuencias

- El repositorio basta para reproducir todo, sin servicios externos.
- Una futura API deberá: recibir solo las variables del modelo (nunca sexo ni raza), registrar cada
  predicción para auditoría y devolver un puntaje que revise un analista, no una aprobación automática.
