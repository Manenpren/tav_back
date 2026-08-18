# TechyNest HydroDam — motor de tránsito y endpoint PHP

Backend del cálculo de tránsito de avenidas en vasos con uno o dos vertedores y obra de toma.

## Motor

El repositorio conserva dos implementaciones:

- API FastAPI para desarrollo/pruebas del motor.
- `php/simulate.php` para el despliegue real en hosting PHP compartido.

La formalización 1.1.0 no modifica las funciones de cálculo del port PHP; añade únicamente acceso, CSRF, configuración y endurecimiento alrededor del motor.

## PHP de producción

Copie `.env.example` como `.env` fuera de `public_html` y configure las variables `HYDRODAM_DB_*`. Por defecto el endpoint:

1. exige cookie `hydro_session`;
2. valida la sesión contra `https://hidro.manserempr.com.mx/api/auth/me`;
3. compara `X-CSRF-Token` con el token de sesión;
4. ejecuta el tránsito;
5. persiste la corrida en la base existente.

Consulte `docs/DEPLOY_SHARED_HOSTING.md` para la estructura pública/privada recomendada.

## Pruebas del motor

```bash
python -m pytest -q
```

Resultado verificado para esta entrega: 18 pruebas aprobadas.

## Endpoints FastAPI de desarrollo

- `POST /simulate`
- `POST /simulate/files`

## Unidades

- elevación: m;
- almacenamiento: m³;
- caudal: m³/s;
- tiempo: h.
