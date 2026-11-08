# ADR-001. Enfoque de implementación: pipeline modular en Python con notebook y pruebas

- **Estado:** aceptada
- **Fecha:** 8 de noviembre de 2026

## Contexto

El proyecto debe mostrar cada etapa del flujo de Machine Learning (datos, preparación, partición,
entrenamiento, evaluación y auditoría ética) y permitir que otra persona reproduzca los resultados.
Un notebook solo es cómodo para explicar, pero difícil de probar y de revisar en Git.

## Decisión

- La lógica vive en módulos de Python (`src/`), uno por etapa: `data.py`, `features.py`,
  `train.py`, `evaluate.py` y `fairness.py`. `src/run_all.py` ejecuta el flujo completo.
- El preprocesamiento va dentro de un `Pipeline` de scikit-learn, para que imputación, escalado y
  codificación se ajusten solo con los datos de entrenamiento.
- El notebook `notebooks/01_flujo_ml_ingreso_justo.ipynb` importa esos módulos y narra cada etapa.
- Los criterios de aceptación son pruebas automáticas con `pytest` (`tests/`).
- Semilla fija (42) en partición, validación cruzada y modelos.

## Alternativas consideradas

| Alternativa | Por qué no |
|---|---|
| Solo notebook | No se puede probar por partes y genera diferencias enormes en Git |
| AutoML o un servicio en la nube (por ejemplo, SageMaker) | Costo, menos control sobre el preprocesamiento y menos transparencia para la auditoría ética |

## Consecuencias

- Cualquiera puede reproducir los números con `python -m src.run_all` y verificar los criterios con `pytest`.
- La misma función se usa en el notebook, en las pruebas y en el script, sin duplicar código.
- Requiere disciplina: un cambio en `src/` obliga a volver a ejecutar el notebook.
