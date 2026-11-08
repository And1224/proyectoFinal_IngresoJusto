# Datos

| Campo | Valor |
|---|---|
| Dataset | Adult (también llamado *Census Income*) |
| Fuente original | UCI Machine Learning Repository: Becker, B. y Kohavi, R. (1996). https://doi.org/10.24432/C5XW20 |
| Licencia | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Archivo | `raw/adult.csv`: archivos de entrenamiento y prueba de UCI combinados, sin encabezado (48 842 filas) |
| Copia obtenida de | Espejo público `jbrownlee/Datasets` en GitHub (`adult-all.csv`) |
| SHA-256 | `21e0ea2f925a00338929a8c86c27354c72ac1c79819bcca81c7d91c3d64218c2` |
| Origen de los registros | Current Population Survey de la Oficina del Censo de EE. UU., 1994 |

## Variables

| Columna | Tipo | Descripción | Uso en el modelo |
|---|---|---|---|
| age | Numérica | Edad en años | Sí |
| workclass | Categórica | Tipo de empleador (privado, gobierno, independiente…) | Sí |
| fnlwgt | Numérica | Peso muestral del censo | No: no describe a la persona |
| education | Categórica | Nivel educativo | No: redundante con education_num |
| education_num | Numérica | Nivel educativo en escala ordinal (1–16) | Sí |
| marital_status | Categórica | Estado civil | Sí |
| occupation | Categórica | Ocupación | Sí |
| relationship | Categórica | Rol en el hogar (Husband, Wife, Own-child…) | Sí (proxy del sexo, ver README) |
| race | Categórica | Raza registrada en el censo | No: atributo protegido, solo auditoría |
| sex | Categórica | Sexo registrado en el censo | No: atributo protegido, solo auditoría |
| capital_gain | Numérica | Ganancias de capital | Sí (log1p) |
| capital_loss | Numérica | Pérdidas de capital | Sí (log1p) |
| hours_per_week | Numérica | Horas trabajadas por semana | Sí |
| native_country | Categórica | País de origen | No: atributo sensible |
| income | Objetivo | `>50K` o `<=50K` | Objetivo (1 = `>50K`) |

Los valores faltantes vienen como `?` en `workclass`, `occupation` y `native_country`.
