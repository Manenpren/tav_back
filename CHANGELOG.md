# Changelog

## 1.1.0 — 2026-08-08

- `php/simulate.php` deja de contener credenciales MySQL embebidas.
- Nueva carga de configuración desde `.env` mediante `php/bootstrap.php`.
- Sesión TechyNest obligatoria por defecto, validada contra `/api/auth/me`.
- Validación de `X-CSRF-Token` antes de ejecutar y persistir una corrida.
- Límite de cuerpo, CORS restringido, encabezados de seguridad y errores internos ocultos en producción.
- Gateway público preparado para mantener el backend y `.env` fuera de `public_html`.
- El bloque de funciones del motor hidráulico se conserva byte a byte respecto al `simulate.php` recibido.
