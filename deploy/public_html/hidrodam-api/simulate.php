<?php
declare(strict_types=1);

// Gateway público. El backend real permanece fuera de public_html.
$backend = dirname(__DIR__, 2) . '/hydrodam-backend/php/simulate.php';
if (!is_file($backend)) {
    http_response_code(503);
    header('Content-Type: application/json; charset=utf-8');
    echo json_encode([
        'error' => [
            'code' => 'HYDRODAM_BACKEND_NOT_DEPLOYED',
            'reason' => 'El backend privado de HydroDam no está instalado en la ruta esperada.',
        ],
    ], JSON_UNESCAPED_UNICODE | JSON_UNESCAPED_SLASHES);
    exit;
}
require $backend;
