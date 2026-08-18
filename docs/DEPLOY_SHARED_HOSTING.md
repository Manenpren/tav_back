# Despliegue en hosting compartido — HydroDam 1.1.0

## Estructura recomendada

La entrega de despliegue está separada en dos zonas:

```text
/home/<cuenta>/
├── hydrodam-backend/
│   ├── .env
│   └── php/
│       ├── bootstrap.php
│       └── simulate.php
└── public_html/
    ├── hidrodam/
    │   └── ...frontend dist...
    └── hidrodam-api/
        ├── .htaccess
        └── simulate.php   # gateway público
```

El gateway usa `dirname(__DIR__, 2)` y por ello funciona sin escribir el nombre de la cuenta del hosting.

## 1. Frontend

Subir el contenido de `dist/` a `/public_html/hidrodam/`.

El frontend espera:

- autenticación central: `/api`
- cálculo HydroDam: `/hidrodam-api/simulate.php`

## 2. Backend privado

Crear `/home/<cuenta>/hydrodam-backend/` fuera de `public_html` y subir:

- `php/bootstrap.php`
- `php/simulate.php`
- `.env` generado a partir de `.env.example`

Configurar en `.env`:

- `HYDRODAM_DB_*`
- `HYDRODAM_AUTH_ME_URL=https://hidro.manserempr.com.mx/api/auth/me`
- `HYDRODAM_SESSION_COOKIE=hydro_session`
- `HYDRODAM_REQUIRE_AUTH=true`
- `HYDRODAM_ALLOWED_ORIGIN=https://hidro.manserempr.com.mx`

## 3. Gateway público

Subir los archivos de `deploy/public_html/hidrodam-api/` a `/public_html/hidrodam-api/`.

No copie `.env` ni credenciales dentro de `public_html`.

## 4. Prueba de humo

1. Abrir `/hidrodam/` en incógnito: debe aparecer la pantalla de acceso.
2. Probar `Entrar como invitado`: debe abrir el proyecto.
3. Cerrar sesión: debe regresar al acceso.
4. Iniciar sesión con un usuario válido.
5. Ejecutar una corrida: no debe responder 401 ni 403.
6. Abrir directamente `simulate.php` sin cookie: debe responder `401 AUTHENTICATION_REQUIRED`.
7. Enviar POST con cookie pero sin `X-CSRF-Token`: debe responder `403 CSRF_VALIDATION_FAILED`.
