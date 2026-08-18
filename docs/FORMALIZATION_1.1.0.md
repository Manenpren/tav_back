# Formalización TechyNest HydroDam 1.1.0

## Objetivo

Alinear HydroDam con el esquema de acceso del ecosistema TechyNest sin modificar el motor de tránsito de avenidas.

## Cambios de seguridad

- Se eliminan credenciales MySQL embebidas de `php/simulate.php`.
- La configuración sensible se lee desde `.env` fuera del código público.
- El endpoint requiere una sesión TechyNest válida por defecto.
- La sesión se verifica contra `/api/auth/me` reutilizando la cookie `hydro_session`.
- Cada `POST /simulate.php` requiere `X-CSRF-Token` y lo compara con el token asociado a la sesión validada.
- Se limita el tamaño del cuerpo JSON y se añaden encabezados de seguridad.
- Los errores de conexión/persistencia dejan de exponer mensajes internos en producción.
- Se mantiene la misma función `simulate()` y las mismas ecuaciones del motor PHP.

## Flujo esperado

1. El navegador entra a `/hidrodam/`.
2. El frontend consulta `/api/auth/me`.
3. Si no existe sesión, permite login, registro o invitado mediante la autenticación central.
4. El navegador recibe la cookie `hydro_session` y el `csrf_token` de sesión.
5. Al calcular, envía el JSON a `/hidrodam-api/simulate.php`, con credenciales y `X-CSRF-Token`.
6. HydroDam valida la sesión con el backend central antes de ejecutar y persistir el cálculo.

## Nota de arquitectura

Esta implementación usa validación HTTP contra `/api/auth/me` porque el backend HydroDam recibido es un endpoint PHP standalone. Una futura API independiente puede migrar a validación directa de `user_sessions`, como el módulo Hidráulica, sin cambiar el contrato del frontend.
