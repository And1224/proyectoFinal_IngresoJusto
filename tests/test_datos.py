"""Pruebas de la etapa de datos: integridad, partición sin fuga y exclusión de atributos sensibles."""
from src import config
from src.data import clean, load_raw, split


def test_carga_completa():
    raw = load_raw()
    assert raw.shape == (48842, 15)


def test_objetivo_binario_y_sin_duplicados():
    X, y = clean(load_raw())
    assert set(y.unique()) == {0, 1}
    assert not X.assign(y=y).duplicated().any()


def test_particion_sin_fuga_y_estratificada():
    X, y = clean(load_raw())
    p = split(X, y)
    idx = {k: set(v[0].index) for k, v in p.items()}
    assert not idx["train"] & idx["test"]
    assert not idx["train"] & idx["val"]
    assert not idx["val"] & idx["test"]
    assert sum(len(v) for v in idx.values()) == len(X)
    for _, yy in p.values():
        assert abs(yy.mean() - y.mean()) < 0.01


def test_atributos_sensibles_fuera_del_modelo():
    assert not set(config.SENSITIVE) & set(config.FEATURES)
    assert "native_country" not in config.FEATURES
