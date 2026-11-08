# ADR-002. Selección del modelo: regresión logística sin atributos protegidos

- **Estado:** aceptada
- **Fecha:** 8 de noviembre de 2026

## Contexto

Se necesita un clasificador binario (ingreso > 50K) que, además de predecir bien, permita explicar
cada decisión, porque el caso de uso (preselección de crédito) afecta oportunidades de las personas.
Se compararon cuatro candidatos con validación cruzada estratificada de 5 pliegues sobre el conjunto
de entrenamiento (umbral 0,5):

| Modelo | Accuracy | F1 | ROC-AUC |
|---|---|---|---|
| Línea base (clase mayoritaria) | 0,761 | 0,000 | 0,500 |
| Regresión logística | 0,845 | 0,643 | 0,899 |
| Árbol de decisión (profundidad 8) | 0,855 | 0,648 | 0,903 |
| KNN (k = 25) | 0,847 | 0,664 | 0,898 |

## Decisión

- Usar **regresión logística** con regularización L2; `C` se elige por validación cruzada (resultó 0,1).
- Elegir el umbral de decisión que maximiza F1 en el conjunto de **validación** (resultó 0,26), nunca en prueba.
- **No** usar `sex`, `race` ni `native_country` como entradas; conservarlos solo para auditar la equidad.

## Alternativas consideradas

| Alternativa | Por qué no |
|---|---|
| KNN | Mejor F1 por 0,02, pero necesita guardar todos los datos personales de entrenamiento para predecir y no explica sus decisiones |
| Árbol de decisión | Desempeño equivalente, pero más inestable ante pequeños cambios en los datos y con explicaciones más difíciles de comunicar al comité de ética |
| Incluir sexo y raza como entradas | Mejora F1 en 0,01, pero aumenta la brecha de igualdad de oportunidades entre sexos de 0,15 a 0,23 |

## Consecuencias

- Los coeficientes muestran qué variables empujan cada decisión, lo que facilita auditar y explicar.
- Excluir los atributos protegidos no elimina el sesgo: `relationship` (Husband/Wife) funciona como
  variable proxy del sexo. Esto se documenta como riesgo y limita el uso del modelo a apoyo humano.
