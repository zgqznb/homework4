import math
import taichi as ti


WIDTH = 960
HEIGHT = 540
ASPECT_RATIO = WIDTH / HEIGHT
FOV = 45.0
EPS = 1e-4
INF = 1e8
SHADOW_BIAS = 5e-3

# 场景常量
CAMERA_POS = ti.Vector([0.0, 0.0, 5.0])
LIGHT_POS = ti.Vector([2.0, 3.0, 4.0])
LIGHT_COLOR = ti.Vector([1.0, 1.0, 1.0])
BACKGROUND = ti.Vector([0.0, 0.13, 0.11])

SPHERE_CENTER = ti.Vector([-1.2, -0.2, 0.0])
SPHERE_RADIUS = 1.2
SPHERE_COLOR = ti.Vector([0.8, 0.1, 0.1])

CONE_APEX = ti.Vector([1.2, 1.2, 0.0])
CONE_AXIS = ti.Vector([0.0, -1.0, 0.0])  # 从顶点指向底面中心
CONE_HEIGHT = 2.6
CONE_RADIUS = 1.2
CONE_COLOR = ti.Vector([0.6, 0.2, 0.8])

DEFAULT_KA = 0.2
DEFAULT_KD = 0.7
DEFAULT_KS = 0.5
DEFAULT_SHININESS = 32.0
DEFAULT_USE_BLINN = 0
DEFAULT_ENABLE_SHADOW = 0


def init_taichi():
    try:
        ti.init(arch=ti.gpu, default_fp=ti.f32)
        print("[Info] Taichi backend: gpu")
    except Exception:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
        print("[Info] Taichi backend: cpu")


init_taichi()

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(WIDTH, HEIGHT))
ka = ti.field(dtype=ti.f32, shape=())
kd = ti.field(dtype=ti.f32, shape=())
ks = ti.field(dtype=ti.f32, shape=())
shininess = ti.field(dtype=ti.f32, shape=())
use_blinn = ti.field(dtype=ti.i32, shape=())
enable_shadow = ti.field(dtype=ti.i32, shape=())


@ti.func
def safe_normalize(v):
    n = v.norm()
    if n < 1e-8:
        return ti.Vector([0.0, 0.0, 0.0])
    return v / n


@ti.func
def clamp01(c):
    return ti.Vector([
        ti.min(1.0, ti.max(0.0, c[0])),
        ti.min(1.0, ti.max(0.0, c[1])),
        ti.min(1.0, ti.max(0.0, c[2])),
    ])


@ti.func
def reflect(i, n):
    return i - 2.0 * i.dot(n) * n


@ti.func
def get_object_color(obj_id):
    color = BACKGROUND
    if obj_id == 1:
        color = SPHERE_COLOR
    elif obj_id == 2:
        color = CONE_COLOR
    return color


@ti.func
def intersect_sphere(ray_o, ray_d):
    hit = 0
    t = INF
    pos = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 0.0, 0.0])

    oc = ray_o - SPHERE_CENTER
    a = ray_d.dot(ray_d)
    b = 2.0 * oc.dot(ray_d)
    c = oc.dot(oc) - SPHERE_RADIUS * SPHERE_RADIUS
    disc = b * b - 4.0 * a * c

    if disc >= 0.0:
        sqrt_disc = ti.sqrt(disc)
        t0 = (-b - sqrt_disc) / (2.0 * a)
        t1 = (-b + sqrt_disc) / (2.0 * a)

        candidate_t = INF
        if t0 > EPS:
            candidate_t = t0
        elif t1 > EPS:
            candidate_t = t1

        if candidate_t < INF:
            hit = 1
            t = candidate_t
            pos = ray_o + t * ray_d
            normal = safe_normalize(pos - SPHERE_CENTER)

    return hit, t, pos, normal


@ti.func
def intersect_cone(ray_o, ray_d):
    hit = 0
    best_t = INF
    pos = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 0.0, 0.0])

    k = CONE_RADIUS / CONE_HEIGHT
    q = 1.0 + k * k
    base_center = CONE_APEX + CONE_AXIS * CONE_HEIGHT

    # 1) 圆锥侧面求交
    co = ray_o - CONE_APEX
    dv = ray_d.dot(CONE_AXIS)
    cv = co.dot(CONE_AXIS)

    a = ray_d.dot(ray_d) - q * dv * dv
    b = 2.0 * (co.dot(ray_d) - q * cv * dv)
    c = co.dot(co) - q * cv * cv

    if ti.abs(a) > 1e-6:
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            sqrt_disc = ti.sqrt(disc)
            t0 = (-b - sqrt_disc) / (2.0 * a)
            t1 = (-b + sqrt_disc) / (2.0 * a)

            if t0 > EPS:
                h0 = cv + t0 * dv
                if 0.0 <= h0 <= CONE_HEIGHT and t0 < best_t:
                    best_t = t0
                    pos = ray_o + best_t * ray_d
                    w = pos - CONE_APEX
                    h = w.dot(CONE_AXIS)
                    normal = safe_normalize(w - q * h * CONE_AXIS)
                    hit = 1

            if t1 > EPS:
                h1 = cv + t1 * dv
                if 0.0 <= h1 <= CONE_HEIGHT and t1 < best_t:
                    best_t = t1
                    pos = ray_o + best_t * ray_d
                    w = pos - CONE_APEX
                    h = w.dot(CONE_AXIS)
                    normal = safe_normalize(w - q * h * CONE_AXIS)
                    hit = 1

    # 2) 底面圆盘求交
    denom = ray_d.dot(CONE_AXIS)
    if ti.abs(denom) > 1e-6:
        t_plane = (base_center - ray_o).dot(CONE_AXIS) / denom
        if t_plane > EPS and t_plane < best_t:
            p = ray_o + t_plane * ray_d
            radial = p - base_center
            radial = radial - radial.dot(CONE_AXIS) * CONE_AXIS
            if radial.norm() <= CONE_RADIUS:
                best_t = t_plane
                pos = p
                normal = CONE_AXIS
                hit = 1

    return hit, best_t, pos, normal


@ti.func
def scene_intersect(ray_o, ray_d):
    hit = 0
    t_min = INF
    pos = ti.Vector([0.0, 0.0, 0.0])
    normal = ti.Vector([0.0, 0.0, 0.0])
    obj_id = 0

    sphere_hit, sphere_t, sphere_pos, sphere_n = intersect_sphere(ray_o, ray_d)
    if sphere_hit == 1 and sphere_t < t_min:
        hit = 1
        t_min = sphere_t
        pos = sphere_pos
        normal = sphere_n
        obj_id = 1

    cone_hit, cone_t, cone_pos, cone_n = intersect_cone(ray_o, ray_d)
    if cone_hit == 1 and cone_t < t_min:
        hit = 1
        t_min = cone_t
        pos = cone_pos
        normal = cone_n
        obj_id = 2

    return hit, t_min, pos, normal, obj_id


@ti.func
def is_occluded(shadow_o, shadow_d, max_dist):
    hit, t, _, _, _ = scene_intersect(shadow_o, shadow_d)
    return hit == 1 and t < max_dist


@ti.kernel
def render():
    fov_scale = ti.tan(FOV * 0.5 * math.pi / 180.0)

    for i, j in pixels:
        x = (2.0 * (i + 0.5) / WIDTH - 1.0) * ASPECT_RATIO * fov_scale
        y = (1.0 - 2.0 * (j + 0.5) / HEIGHT) * fov_scale

        pixel_pos = ti.Vector([x, y, 0.0])
        ray_o = CAMERA_POS
        ray_d = safe_normalize(pixel_pos - CAMERA_POS)

        color = BACKGROUND
        hit, _, hit_pos, normal, obj_id = scene_intersect(ray_o, ray_d)

        if hit == 1:
            obj_color = get_object_color(obj_id)
            l = safe_normalize(LIGHT_POS - hit_pos)
            v = safe_normalize(CAMERA_POS - hit_pos)

            ambient = ka[None] * LIGHT_COLOR * obj_color

            ndotl = ti.max(0.0, normal.dot(l))
            diffuse = kd[None] * ndotl * LIGHT_COLOR * obj_color

            specular_factor = 0.0
            if ndotl > 0.0:
                if use_blinn[None] == 1:
                    h = safe_normalize(l + v)
                    specular_factor = ti.pow(ti.max(0.0, normal.dot(h)), shininess[None])
                else:
                    r = safe_normalize(reflect(-l, normal))
                    specular_factor = ti.pow(ti.max(0.0, r.dot(v)), shininess[None])

            specular = ks[None] * specular_factor * LIGHT_COLOR

            if enable_shadow[None] == 1:
                shadow_origin = hit_pos + normal * SHADOW_BIAS
                light_vec = LIGHT_POS - hit_pos
                light_dist = light_vec.norm()
                if is_occluded(shadow_origin, l, light_dist - SHADOW_BIAS):
                    diffuse = ti.Vector([0.0, 0.0, 0.0])
                    specular = ti.Vector([0.0, 0.0, 0.0])

            color = ambient + diffuse + specular

        pixels[i, j] = clamp01(color)


def reset_params():
    ka[None] = DEFAULT_KA
    kd[None] = DEFAULT_KD
    ks[None] = DEFAULT_KS
    shininess[None] = DEFAULT_SHININESS
    use_blinn[None] = DEFAULT_USE_BLINN
    enable_shadow[None] = DEFAULT_ENABLE_SHADOW


def run():
    window = ti.ui.Window("Phong / Blinn-Phong Ray Casting Demo", (WIDTH, HEIGHT), vsync=True)
    canvas = window.get_canvas()
    gui = window.get_gui()

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == ti.ui.ESCAPE:
                break

        with gui.sub_window("Material Parameters", 0.53, 0.05, 0.42, 0.40):
            gui.text("Core parameters")
            ka[None] = gui.slider_float("Ka (Ambient)", ka[None], 0.0, 1.0)
            kd[None] = gui.slider_float("Kd (Diffuse)", kd[None], 0.0, 1.0)
            ks[None] = gui.slider_float("Ks (Specular)", ks[None], 0.0, 1.0)
            shininess[None] = gui.slider_float("Shininess", shininess[None], 1.0, 128.0)

            gui.text("Optional features")
            blinn_enabled = gui.checkbox("Enable Blinn-Phong", use_blinn[None] == 1)
            shadow_enabled = gui.checkbox("Enable Hard Shadow", enable_shadow[None] == 1)
            use_blinn[None] = 1 if blinn_enabled else 0
            enable_shadow[None] = 1 if shadow_enabled else 0

            if gui.button("Reset to defaults"):
                reset_params()

            mode_name = "Blinn-Phong" if use_blinn[None] == 1 else "Phong"
            shadow_name = "ON" if enable_shadow[None] == 1 else "OFF"
            gui.text(f"Specular mode: {mode_name}")
            gui.text(f"Hard shadow: {shadow_name}")
            gui.text("ESC: quit")

        render()
        canvas.set_image(pixels)
        window.show()


if __name__ == "__main__":
    reset_params()
    run()
