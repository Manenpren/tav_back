<?php
declare(strict_types=1);

function hydrodam_load_env(string $path): void {
    if (!is_file($path) || !is_readable($path)) return;
    $lines = file($path, FILE_IGNORE_NEW_LINES | FILE_SKIP_EMPTY_LINES);
    if ($lines === false) return;
    foreach ($lines as $line) {
        $line = trim($line);
        if ($line === '' || str_starts_with($line, '#') || !str_contains($line, '=')) continue;
        [$key, $value] = explode('=', $line, 2);
        $key = trim($key);
        $value = trim($value);
        if ($key === '' || getenv($key) !== false) continue;
        if ((str_starts_with($value, '"') && str_ends_with($value, '"')) || (str_starts_with($value, "'") && str_ends_with($value, "'"))) {
            $value = substr($value, 1, -1);
        }
        putenv($key . '=' . $value);
        $_ENV[$key] = $value;
    }
}

hydrodam_load_env(getenv('HYDRODAM_ENV_FILE') ?: dirname(__DIR__) . '/.env');

function hydrodam_env(string $key, ?string $default = null): ?string {
    $value = getenv($key);
    return $value === false ? $default : $value;
}

function hydrodam_bool(string $key, bool $default = false): bool {
    $value = hydrodam_env($key);
    if ($value === null || $value === '') return $default;
    return in_array(strtolower($value), ['1', 'true', 'yes', 'on'], true);
}

function hydrodam_json(array $payload, int $status = 200): never {
    http_response_code($status);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode($payload, JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    exit;
}

function hydrodam_security_headers(): void {
    header('X-Content-Type-Options: nosniff');
    header('Referrer-Policy: strict-origin-when-cross-origin');
    header('Cache-Control: no-store');
    $allowed = hydrodam_env('HYDRODAM_ALLOWED_ORIGIN', 'https://hidro.manserempr.com.mx');
    $origin = $_SERVER['HTTP_ORIGIN'] ?? '';
    if ($allowed && $origin === $allowed) {
        header('Access-Control-Allow-Origin: ' . $allowed);
        header('Access-Control-Allow-Credentials: true');
        header('Vary: Origin');
    }
    header('Access-Control-Allow-Headers: Content-Type, X-CSRF-Token');
    header('Access-Control-Allow-Methods: POST, OPTIONS');
}

function hydrodam_request_header(string $name): ?string {
    $serverKey = 'HTTP_' . strtoupper(str_replace('-', '_', $name));
    if (isset($_SERVER[$serverKey])) return trim((string) $_SERVER[$serverKey]);
    if (function_exists('getallheaders')) {
        foreach (getallheaders() as $key => $value) {
            if (strcasecmp($key, $name) === 0) return trim((string) $value);
        }
    }
    return null;
}

function hydrodam_fetch_auth_session(string $url, string $cookieName, string $cookieValue): array {
    $timeout = max(1, (int) hydrodam_env('HYDRODAM_AUTH_TIMEOUT_SECONDS', '5'));
    $userAgent = $_SERVER['HTTP_USER_AGENT'] ?? 'TechyNest-HydroDam/1.1';
    $headers = [
        'Accept: application/json',
        'Cookie: ' . $cookieName . '=' . rawurlencode($cookieValue),
        'User-Agent: ' . $userAgent,
    ];

    $status = 0;
    $body = false;
    if (function_exists('curl_init')) {
        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_FOLLOWLOCATION => false,
            CURLOPT_CONNECTTIMEOUT => $timeout,
            CURLOPT_TIMEOUT => $timeout,
            CURLOPT_HTTPHEADER => $headers,
            CURLOPT_SSL_VERIFYPEER => true,
        ]);
        $body = curl_exec($ch);
        $status = (int) curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
        curl_close($ch);
    } else {
        $context = stream_context_create([
            'http' => [
                'method' => 'GET',
                'header' => implode("\r\n", $headers),
                'timeout' => $timeout,
                'ignore_errors' => true,
            ],
        ]);
        $body = @file_get_contents($url, false, $context);
        foreach (($http_response_header ?? []) as $line) {
            if (preg_match('/^HTTP\/\S+\s+(\d{3})/', $line, $match)) {
                $status = (int) $match[1];
                break;
            }
        }
    }

    if (!is_string($body) || $status < 200 || $status >= 300) {
        throw new RuntimeException('AUTHENTICATION_REQUIRED');
    }
    $payload = json_decode($body, true);
    if (!is_array($payload)) throw new RuntimeException('AUTHENTICATION_REQUIRED');
    $session = isset($payload['data']) && is_array($payload['data']) ? $payload['data'] : $payload;
    if (!isset($session['user_id']) && !isset($session['email']) && !isset($session['display_name'])) {
        throw new RuntimeException('AUTHENTICATION_REQUIRED');
    }
    return $session;
}

function hydrodam_require_session(): array {
    if (!hydrodam_bool('HYDRODAM_REQUIRE_AUTH', true)) {
        return ['user_id' => null, 'client_id' => null, 'display_name' => 'Auth disabled'];
    }

    $cookieName = hydrodam_env('HYDRODAM_SESSION_COOKIE', 'hydro_session') ?: 'hydro_session';
    $cookieValue = $_COOKIE[$cookieName] ?? null;
    if (!is_string($cookieValue) || $cookieValue === '') {
        hydrodam_json(['error' => ['code' => 'AUTHENTICATION_REQUIRED', 'reason' => 'Inicia sesión o usa el modo invitado para calcular.']], 401);
    }

    try {
        $session = hydrodam_fetch_auth_session(
            hydrodam_env('HYDRODAM_AUTH_ME_URL', 'https://hidro.manserempr.com.mx/api/auth/me') ?: '',
            $cookieName,
            $cookieValue
        );
    } catch (Throwable $e) {
        hydrodam_json(['error' => ['code' => 'AUTHENTICATION_REQUIRED', 'reason' => 'La sesión TechyNest no está activa.']], 401);
    }

    $requestCsrf = hydrodam_request_header('X-CSRF-Token');
    $sessionCsrf = isset($session['csrf_token']) ? (string) $session['csrf_token'] : '';
    if ($sessionCsrf === '' || $requestCsrf === null || !hash_equals($sessionCsrf, $requestCsrf)) {
        hydrodam_json(['error' => ['code' => 'CSRF_VALIDATION_FAILED', 'reason' => 'La solicitud no contiene un token CSRF válido.']], 403);
    }
    return $session;
}

function hydrodam_database(): mysqli {
    mysqli_report(MYSQLI_REPORT_ERROR | MYSQLI_REPORT_STRICT);
    $host = hydrodam_env('HYDRODAM_DB_HOST', 'localhost') ?: 'localhost';
    $port = (int) (hydrodam_env('HYDRODAM_DB_PORT', '3306') ?: '3306');
    $name = hydrodam_env('HYDRODAM_DB_NAME');
    $user = hydrodam_env('HYDRODAM_DB_USER');
    $password = hydrodam_env('HYDRODAM_DB_PASSWORD', '') ?? '';
    if (!$name || !$user) throw new RuntimeException('DATABASE_CONFIGURATION_MISSING');
    $conn = new mysqli($host, $user, $password, $name, $port);
    $conn->set_charset('utf8mb4');
    return $conn;
}
