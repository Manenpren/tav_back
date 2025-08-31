<?php
header('Content-Type: application/json');

// Database connection
$db_host="localhost"; // Nombre del host
$db_user="manserem_AdminCo"; // Usuario de la base de datos
$db_pass=""; // Contraseña de usuario de base de datos
$db_name="manserem_tav"; // Nombre de la base de datos
$conn = new mysqli($db_host, $db_user, $db_pass, $db_name);
if ($conn->connect_error) {
    http_response_code(500);
    echo json_encode(["error" => "Conexión falló: " . $conn->connect_error]);
    exit;
}

if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    echo json_encode(["error" => "Método no soportado. Usa POST con JSON"]);
    exit;
}

$input = file_get_contents('php://input');
$data = json_decode($input, true);
if (!$data) {
    http_response_code(400);
    echo json_encode(["error" => "JSON inválido"]);
    exit;
}

try {
    list($results, $summary) = simulate($data);
} catch (Exception $e) {
    http_response_code(422);
    echo json_encode(["error" => $e->getMessage()]);
    exit;
}

// ---- Store in DB ----
try {
    $conn->begin_transaction();

    $description = isset($data['description']) ? "'".$conn->real_escape_string($data['description'])."'" : "NULL";
    $gravity = floatval($data['gravity']);
    $tolerance = floatval($data['tolerance']);
    $max_iter_step = intval($data['max_iter_step']);
    $dt_hours = $summary['dt_hours'];
    $velocity_head_enabled = !empty($data['velocity_head_enabled']) ? 1 : 0;
    $initial_level_m = isset($data['initial_level_m']) ? floatval($data['initial_level_m']) : "NULL";
    $initial_volume_m3 = isset($data['initial_volume_m3']) ? floatval($data['initial_volume_m3']) : "NULL";
    $drain_tail = !empty($data['drain_tail']) ? 1 : 0;
    $tail_margin_m = floatval($data['tail_margin_m']);
    $tail_hours_limit = isset($data['tail_hours_limit']) ? floatval($data['tail_hours_limit']) : "NULL";

    $max_level_m = $summary['max_level_m'];
    $max_stored_hm3 = $summary['max_stored_hm3'];
    $peak_inflow_m3s = $summary['peak_inflow_m3s'];
    $t_peak_inflow_h = $summary['t_peak_inflow_h'];
    $peak_outflow_m3s = $summary['peak_outflow_m3s'];
    $t_peak_outflow_h = $summary['t_peak_outflow_h'];
    $peak_q1_m3s = $summary['peak_q1_m3s'];
    $peak_q2_m3s = $summary['peak_q2_m3s'];
    $coeff1_at_peak = $summary['coeff1_at_peak'];
    $coeff2_at_peak = $summary['coeff2_at_peak'];

    $sql = "INSERT INTO simulations (description, gravity, tolerance, max_iter_step, dt_hours, velocity_head_enabled, initial_level_m, initial_volume_m3, drain_tail, tail_margin_m, tail_hours_limit, max_level_m, max_stored_hm3, peak_inflow_m3s, t_peak_inflow_h, peak_outflow_m3s, t_peak_outflow_h, peak_q1_m3s, peak_q2_m3s, coeff1_at_peak, coeff2_at_peak) VALUES ($description,$gravity,$tolerance,$max_iter_step,$dt_hours,$velocity_head_enabled,$initial_level_m,$initial_volume_m3,$drain_tail,$tail_margin_m,$tail_hours_limit,$max_level_m,$max_stored_hm3,$peak_inflow_m3s,$t_peak_inflow_h,$peak_outflow_m3s,$t_peak_outflow_h,$peak_q1_m3s,$peak_q2_m3s,$coeff1_at_peak,$coeff2_at_peak)";
    if (!$conn->query($sql)) throw new Exception($conn->error);
    $simulation_id = $conn->insert_id;

    // ELCAP
    if (isset($data['elev_volume'])) {
        foreach ($data['elev_volume'] as $pt) {
            $e = floatval($pt['elevation_m']);
            $v = floatval($pt['volume_m3']);
            $conn->query("INSERT INTO simulation_elcap (simulation_id, elevation_m, volume_m3) VALUES ($simulation_id,$e,$v)");
        }
    }

    // INFLOW
    if (isset($data['inflow'])) {
        foreach ($data['inflow'] as $pt) {
            $t = floatval($pt['t_hours']);
            $q = floatval($pt['q_m3s']);
            $conn->query("INSERT INTO simulation_inflow (simulation_id, t_hours, q_m3s) VALUES ($simulation_id,$t,$q)");
        }
    }

    // Spillways
    for ($sp=1; $sp<=2; $sp++) {
        $spkey = $sp==1 ? 'spillway1' : 'spillway2';
        if (!isset($data[$spkey])) continue;
        $cfg = $data[$spkey];
        $number = $sp;
        $mode = $conn->real_escape_string($cfg['mode']);
        $crest = floatval($cfg['crest_m']);
        $design_head = floatval($cfg['design_head_m']);
        $length = floatval($cfg['length_m']);
        $approach = floatval($cfg['approach_depth_m']);
        $discharge_coefficient = isset($cfg['discharge_coefficient']) ? floatval($cfg['discharge_coefficient']) : "NULL";
        $auto_coefficient = !empty($cfg['auto_coefficient']) ? 1 : 0;
        $slope_correction_enabled = !empty($cfg['slope_correction_enabled']) ? 1 : 0;
        $slope_variant = intval($cfg['slope_variant']);
        $els_enabled = !empty($cfg['els_enabled']) ? 1 : 0;
        $conn->query("INSERT INTO simulation_spillways (simulation_id, number, mode, crest_m, design_head_m, length_m, approach_depth_m, discharge_coefficient, auto_coefficient, slope_correction_enabled, slope_variant, els_enabled) VALUES ($simulation_id,$number,'$mode',$crest,$design_head,$length,$approach,$discharge_coefficient,$auto_coefficient,$slope_correction_enabled,$slope_variant,$els_enabled)");
    }

    // Steps
    foreach ($results as $step) {
        $t = $step['t_hours'];
        $inflow = $step['inflow_m3s'];
        $level = $step['level_m'];
        $volume = $step['volume_m3'];
        $q1 = $step['q1_m3s'];
        $q2 = $step['q2_m3s'];
        $q_intake = $step['q_intake_m3s'];
        $q_total = $step['q_total_m3s'];
        $c1 = $step['coeff1'];
        $c2 = $step['coeff2'];
        $vh1 = $step['vel_head1'];
        $vh2 = $step['vel_head2'];
        $conn->query("INSERT INTO simulation_steps (simulation_id, t_hours, inflow_m3s, level_m, volume_m3, q1_m3s, q2_m3s, q_intake_m3s, q_total_m3s, coeff1, coeff2, vel_head1, vel_head2) VALUES ($simulation_id,$t,$inflow,$level,$volume,$q1,$q2,$q_intake,$q_total,$c1,$c2,$vh1,$vh2)");
    }

    $conn->commit();
} catch (Exception $e) {
    $conn->rollback();
    http_response_code(500);
    echo json_encode(["error" => $e->getMessage()]);
    exit;
}

echo json_encode(["summary" => $summary, "timeseries" => $results]);

// ---------------- Simulation Functions ----------------
function interp_y_from_x($xs, $ys, $x) {
    $n = count($xs);
    if ($n < 2) return null;
    $paired = [];
    for ($i=0; $i<$n; $i++) { $paired[] = [$xs[$i], $ys[$i]]; }
    usort($paired, function($a, $b) { return $a[0] <=> $b[0]; });
    $xs = array_column($paired, 0);
    $ys = array_column($paired, 1);
    for ($i=1; $i<$n; $i++) {
        if ($xs[$i] <= $xs[$i-1]) $xs[$i] = $xs[$i-1] + 1e-12;
    }
    if ($x < $xs[0] || $x > $xs[$n-1]) return null;
    $j = 0;
    while ($j < $n && $xs[$j] < $x) $j++;
    if ($j == 0) return $ys[0];
    if ($j >= $n) return $ys[$n-1];
    $x1 = $xs[$j-1]; $x2 = $xs[$j];
    $y1 = $ys[$j-1]; $y2 = $ys[$j];
    $w = ($x - $x1)/($x2 - $x1);
    return $y1 + $w*($y2 - $y1);
}

function level_from_volume($elev_volume, $volume) {
    $vol = array_map(fn($p)=>$p['volume_m3'], $elev_volume);
    $lev = array_map(fn($p)=>$p['elevation_m'], $elev_volume);
    return interp_y_from_x($vol, $lev, $volume);
}

function volume_from_level($elev_volume, $level) {
    $vol = array_map(fn($p)=>$p['volume_m3'], $elev_volume);
    $lev = array_map(fn($p)=>$p['elevation_m'], $elev_volume);
    return interp_y_from_x($lev, $vol, $level);
}

function q_from_level($curve, $level) {
    if (!$curve) return null;
    $x = array_map(fn($p)=>$p['elevation_m'], $curve);
    $y = array_map(fn($p)=>$p['q_m3s'], $curve);
    return interp_y_from_x($x, $y, $level);
}

function factor_from_level($curve, $level) {
    if (!$curve) return null;
    $x = array_map(fn($p)=>$p['elevation_m'], $curve);
    $y = array_map(fn($p)=>$p['factor'], $curve);
    return interp_y_from_x($x, $y, $level);
}

function calccT($P_over_Ho, $variant) {
    $x = $P_over_Ho;
    if ($variant == 1) {
        return 0.0000192915*$x**5 + 0.0000652576*$x**4 - 0.0037167326*$x**3 + 0.0139414817*$x**2 - 0.020983809*$x + 1.0130887975;
    } elseif ($variant == 2) {
        return -0.0007284603*$x**5 + 0.0055058805*$x**4 - 0.0224162021*$x**3 + 0.0497832864*$x**2 - 0.0633607576*$x + 1.0372571395;
    } elseif ($variant == 3) {
        return 0.0192355971*$x**6 - 0.1190570309*$x**5 + 0.30854246*$x**4 - 0.4449669487*$x**3 + 0.4010850905*$x**2 - 0.23510752*$x + 1.0688143858;
    }
    return 1.0;
}

function calccn($P_over_Ho) {
    $x = $P_over_Ho;
    if ($x <= 0.4692) {
        return 59.3158102091401*$x**5 - 79.563679870761*$x**4 + 40.7822465493219*$x**3 - 11.3206744409882*$x**2 + 2.51214938629264*$x + 1.70025038012531;
    } else {
        return 0.00664612505920559*$x**5 - 0.0667824975071266*$x**4 + 0.261529474920016*$x**3 - 0.507105271566364*$x**2 + 0.512670950912138*$x + 1.93997699308661;
    }
}

function weir_Q($cfg, $level, $g, $velocity_head_enabled) {
    $crest = $cfg['crest_m'];
    $L = $cfg['length_m'];
    $P = $cfg['approach_depth_m'];
    $Ho = max($cfg['design_head_m'], 1e-9);

    if (!empty($cfg['auto_coefficient']) || !isset($cfg['discharge_coefficient'])) {
        $base_C = calccn($P / $Ho);
    } else {
        $base_C = $cfg['discharge_coefficient'];
    }

    if (!empty($cfg['slope_correction_enabled'])) {
        $CT = calccT($P / $Ho, $cfg['slope_variant']);
        $base_C *= $CT;
    }

    $segment_Q = function($segment_crest, $segment_length) use ($cfg, $level, $g, $P, $base_C, $velocity_head_enabled) {
        $Ha = 0.0; $C_eff = $base_C;
        if ($level <= $segment_crest) return [0.0, 0.0];
        if (!empty($cfg['els_enabled']) && isset($cfg['els_curve'])) {
            $fac = factor_from_level($cfg['els_curve'], $level);
            if ($fac === null) throw new Exception("ELS factor missing for current level (spillway).");
            $C_eff *= $fac;
        }
        for ($i=0; $i<100; $i++) {
            $H = max($level - $segment_crest + $Ha, 0.0);
            $Q = $C_eff * $segment_length * ($H ** 1.5);
            if ($velocity_head_enabled && ($P + $level - $segment_crest) > 0 && $segment_length > 0) {
                $v = $Q / ($segment_length * ($P + $level - $segment_crest));
                $Ha_new = ($v*$v)/(2.0*$g);
            } else {
                $Ha_new = 0.0;
            }
            if (abs($Ha_new - $Ha) < 1e-10) { $Ha = $Ha_new; break; }
            $Ha = $Ha_new;
        }
        return [$Q, $Ha];
    };

    if ($cfg['mode'] === 'CONTROLADO') {
        $q = q_from_level($cfg['policy_q'] ?? null, $level);
        if ($q === null) return [0.0, 0.0, 0.0];
        return [$q, $base_C, 0.0];
    } elseif ($cfg['mode'] === 'CON_AGUJAS') {
        if (empty($cfg['needles'])) return [0.0, $base_C, 0.0];
        $total_q = 0.0; $last_Ha = 0.0;
        foreach ($cfg['needles'] as $seg) {
            list($qi, $hai) = $segment_Q($seg['crest_m'], $seg['length_m']);
            $total_q += $qi; $last_Ha = $hai;
        }
        return [$total_q, $base_C, $last_Ha];
    } else {
        list($q, $ha) = $segment_Q($crest, $L);
        return [$q, $base_C, $ha];
    }
}

function intake_Q($intake, $level) {
    $mode = $intake['mode'] ?? 'OFF';
    if ($mode === 'OFF') return 0.0;
    if ($mode === 'CONSTANT') return floatval($intake['q_constant_m3s'] ?? 0.0);
    if ($mode === 'TABLE') {
        $q = q_from_level($intake['q_vs_level'] ?? null, $level);
        return $q !== null ? $q : 0.0;
    }
    return 0.0;
}

function uniform_dt_from_inflow($times) {
    if (count($times) < 2) return 0.0;
    $dts = [];
    for ($i=1; $i<count($times); $i++) $dts[] = $times[$i] - $times[$i-1];
    $first = $dts[0];
    foreach ($dts as $dt) {
        if (abs($dt - $first) > 1e-9) throw new Exception('Inflow time step dt must be uniform.');
    }
    return $first;
}

function simulate($req) {
    $g = $req['gravity'];

    $elev_volume = $req['elev_volume'];
    usort($elev_volume, fn($a,$b)=>$a['volume_m3'] <=> $b['volume_m3']);
    if (count($elev_volume) < 2) throw new Exception('elev_volume requires at least 2 points.');

    $inflow = $req['inflow'];
    usort($inflow, fn($a,$b)=>$a['t_hours'] <=> $b['t_hours']);
    $t = array_map(fn($p)=>$p['t_hours'], $inflow);
    $q_in = array_map(fn($p)=>$p['q_m3s'], $inflow);
    if (isset($req['dt_hours']) && $req['dt_hours'] !== null) {
        $dt_h = $req['dt_hours'];
    } else {
        $dt_h = uniform_dt_from_inflow($t);
    }
    if ($dt_h <= 0) throw new Exception('Invalid dt_hours.');

    if (isset($req['initial_volume_m3'])) {
        $V = $req['initial_volume_m3'];
        $E = level_from_volume($elev_volume, $V);
        if ($E === null) throw new Exception('Initial volume is out of ELCAP bounds.');
    } elseif (isset($req['initial_level_m'])) {
        $E = $req['initial_level_m'];
        $V = volume_from_level($elev_volume, $E);
        if ($V === null) throw new Exception('Initial level is out of ELCAP bounds.');
    } else {
        $E = min(array_map(fn($p)=>$p['elevation_m'], $elev_volume));
        $V = volume_from_level($elev_volume, $E);
        if ($V === null) $V = min(array_map(fn($p)=>$p['volume_m3'], $elev_volume));
    }

    $sp1 = $req['spillway1'];
    $sp2 = $req['spillway2'] ?? null;

    $results = [];

    $outflows = function($level) use ($sp1, $sp2, $req, $g) {
        list($q1, $c1, $ha1) = weir_Q($sp1, $level, $g, !empty($req['velocity_head_enabled']));
        $q2 = 0.0; $c2 = 0.0; $ha2 = 0.0;
        if ($sp2) {
            list($q2, $c2, $ha2) = weir_Q($sp2, $level, $g, !empty($req['velocity_head_enabled']));
        }
        $q_ot = intake_Q($req['intake'] ?? ['mode'=>'OFF'], $level);
        $q_tot = $q1 + $q2 + $q_ot;
        return [$q1, $q2, $q_ot, $q_tot, $c1>0?$c1:0.0, $c2>0?$c2:0.0, $ha1, $ha2];
    };

    $V_prev = $V;
    $level_init = level_from_volume($elev_volume, $V_prev) ?? 0.0;
    list($q1,$q2,$q_ot,$q_tot,$c1,$c2,$ha1,$ha2) = $outflows($level_init);
    $q_prev_total = $q_tot;
    $t0 = $t[0];

    for ($i=1; $i<count($t); $i++) {
        $t1 = $t[$i-1];
        $t2 = $t[$i];
        $I1 = $q_in[$i-1];
        $I2 = $q_in[$i];
        $V_guess = $V_prev + 0.5*(($I1+$I2) - 2*$q_prev_total)*($dt_h*3600.0);
        $minV = min(array_map(fn($p)=>$p['volume_m3'], $elev_volume));
        if ($V_guess < $minV) $V_guess = $minV;
        for ($k=0; $k<$req['max_iter_step']; $k++) {
            $level = level_from_volume($elev_volume, $V_guess);
            if ($level === null) throw new Exception('Volume went out of ELCAP bounds during iteration.');
            list($q1,$q2,$q_ot,$q_tot,$c1,$c2,$ha1,$ha2) = $outflows($level);
            $V_next = $V_prev + 0.5*(($I1+$I2) - ($q_prev_total+$q_tot))*($dt_h*3600.0);
            if ($V_next < $minV) $V_next = $minV;
            if (abs($V_next - $V_guess) < $req['tolerance']) { $V_guess = $V_next; break; }
            $V_guess = $V_next;
        }
        $V_prev = $V_guess;
        $level = level_from_volume($elev_volume, $V_prev) ?? 0.0;
        list($q1,$q2,$q_ot,$q_tot,$c1,$c2,$ha1,$ha2) = $outflows($level);
        $q_prev_total = $q_tot;
        $results[] = [
            't_hours'=>$t2,'inflow_m3s'=>$I2,'level_m'=>$level,'volume_m3'=>$V_prev,
            'q1_m3s'=>$q1,'q2_m3s'=>$q2,'q_intake_m3s'=>$q_ot,'q_total_m3s'=>$q_tot,
            'coeff1'=>$c1,'coeff2'=>$c2,'vel_head1'=>$ha1,'vel_head2'=>$ha2
        ];
    }

    if (!empty($req['drain_tail'])) {
        $crest_min = $sp2 ? min($sp1['crest_m'], $sp2['crest_m']) : $sp1['crest_m'];
        $hours_accum = 0.0;
        while (true) {
            $level = level_from_volume($elev_volume, $V_prev) ?? 0.0;
            if ($level <= $crest_min + $req['tail_margin_m']) break;
            list($q1,$q2,$q_ot,$q_tot,$c1,$c2,$ha1,$ha2) = $outflows($level);
            $V_next = $V_prev - ($q_tot)*($dt_h*3600.0);
            $minV = min(array_map(fn($p)=>$p['volume_m3'], $elev_volume));
            if ($V_next < $minV) break;
            $V_prev = $V_next;
            $t0 += $dt_h;
            $hours_accum += $dt_h;
            $results[] = [
                't_hours'=>$t0,'inflow_m3s'=>0.0,'level_m'=>$level,'volume_m3'=>$V_prev,
                'q1_m3s'=>$q1,'q2_m3s'=>$q2,'q_intake_m3s'=>$q_ot,'q_total_m3s'=>$q_tot,
                'coeff1'=>$c1,'coeff2'=>$c2,'vel_head1'=>$ha1,'vel_head2'=>$ha2
            ];
            if (isset($req['tail_hours_limit']) && $req['tail_hours_limit'] !== null && $hours_accum >= $req['tail_hours_limit']) break;
        }
    }

    $max_level = 0.0; $max_vol = 0.0;
    foreach ($results as $r) {
        if ($r['level_m'] > $max_level) $max_level = $r['level_m'];
        if ($r['volume_m3'] > $max_vol) $max_vol = $r['volume_m3'];
    }
    if (empty($results)) {
        $max_level = level_from_volume($elev_volume, $V) ?? 0.0;
        $max_vol = $V;
    }
    $hm3 = $max_vol / 1e6;
    $peak_inflow_idx = array_keys($q_in, max($q_in))[0];
    $peak_inflow = $q_in[$peak_inflow_idx];
    $t_peak_inflow = $t[$peak_inflow_idx];
    $out_tot = array_map(fn($r)=>$r['q_total_m3s'], $results);
    $peak_outflow_idx = array_keys($out_tot, max($out_tot))[0] ?? 0;
    $peak_outflow = $out_tot[$peak_outflow_idx] ?? 0.0;
    $t_peak_outflow = $results[$peak_outflow_idx]['t_hours'] ?? end($t);
    $peak_q1 = 0.0; $peak_q2 = 0.0; $coeff1_at_peak=0.0; $coeff2_at_peak=0.0;
    foreach ($results as $r) {
        if ($r['q1_m3s'] > $peak_q1) $peak_q1 = $r['q1_m3s'];
        if ($r['q2_m3s'] > $peak_q2) $peak_q2 = $r['q2_m3s'];
    }
    if (!empty($results)) {
        $coeff1_at_peak = $results[$peak_outflow_idx]['coeff1'];
        $coeff2_at_peak = $results[$peak_outflow_idx]['coeff2'];
    }

    $summary = [
        'dt_hours'=>$dt_h,
        'max_level_m'=>$max_level,
        'max_stored_hm3'=>$hm3,
        'peak_inflow_m3s'=>$peak_inflow,
        't_peak_inflow_h'=>$t_peak_inflow,
        'peak_outflow_m3s'=>$peak_outflow,
        't_peak_outflow_h'=>$t_peak_outflow,
        'peak_q1_m3s'=>$peak_q1,
        'peak_q2_m3s'=>$peak_q2,
        'coeff1_at_peak'=>$coeff1_at_peak,
        'coeff2_at_peak'=>$coeff2_at_peak
    ];

    return [$results, $summary];
}
