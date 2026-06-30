import numpy as np
from components.camera import Camera
from components.material import DiffuseMaterial
from components.scene import Scene
from components.shapes import Sphere, Plane, Box
from components.light import AreaLight
from components.film import Film
from components.transform import Transform
from components.instance import Instance
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

scene = Scene()
scene.max_depth = 10
scene.min_depth = 5
scene.rr_probability = 0.95

# =========================
# MATERIAIS
# =========================

# Material vermelho difuso
red_material = DiffuseMaterial(
    albedo=[0.7, 0.1, 0.1]
)

# Material amarelo difuso
yellow_material = DiffuseMaterial(
    albedo=[0.7, 0.7, 0.1]
)

# Material azul difuso
blue_material = DiffuseMaterial(
    albedo=[0.1, 0.1, 0.7]
)

# Material cinza difuso (piso)
gray_material = DiffuseMaterial(
    albedo=[0.6, 0.6, 0.6]
)

# Material verde difuso (fundo)
green_material = DiffuseMaterial(
    albedo=[0.2, 0.5, 0.2]
)

# =========================
# ESFERAS
# =========================
sphere1 = Instance(
    Sphere(
        radius=0.5,
        center=[-1.2, 0.5, -3.5]
    ),
    red_material,
)
scene.add_object(sphere1)

sphere2 = Instance(
    Sphere(
        radius=0.4,
        center=[0.5, 0.4, -3.2]
    ),
    yellow_material,
)
scene.add_object(sphere2)

# =========================
# CAIXA
# =========================
box = Instance(
    shape=Box(
        min_corner=[-0.5,-0.5,-0.5],
        max_corner=[0.5,0.5,0.5]
    ),
    material=blue_material,
    transform=Transform(
        translation=[1.5, 0.5, -3.5],
        rotation=[15, 35, 20],
    )
)
scene.add_object(box)

# =========================
# PLANO CHAO
# =========================
floor = Instance(
    Plane(
        normal=[0, 1, 0],
        material=gray_material
    ),
    material=gray_material,
    transform=Transform(
        translation=[0, 0, 0]
    )
)
scene.add_object(floor)

# =========================
# PLANO FUNDO
# =========================
back_plane = Instance(
    Plane(
        normal=[0, 0, 1],
        material=green_material
    ),
    material=green_material,
    transform=Transform(
        translation=[0, 2, -5]
    )
)
scene.add_object(back_plane)

# =========================
# PAREDE ESQUERDA
# =========================
left_wall = Instance(
    Box(
        min_corner=[-0.01, 0.0, -6.0],
        max_corner=[ 0.01, 4.0,  0.0],
        material=blue_material
    ),
    material=blue_material,
    transform=Transform(
        translation=[-3.0, 0.0, 0.0]
    )
)
scene.add_object(left_wall)

# =========================
# PAREDE DIREITA
# =========================
right_wall = Instance(
    Box(
        min_corner=[-0.01, 0.0, -6.0],
        max_corner=[ 0.01, 4.0,  0.0],
        material=red_material
    ),
    material=red_material,
    transform=Transform(
        translation=[3.0, 0.0, 0.0]
    )
)
scene.add_object(right_wall)

# =========================
# LUZES RETANGULARES
# =========================
main_light = AreaLight(
    position=[0, 3.5, -3],
    normal=[0, -1, 0],
    width=2.0,
    height=2.0,
    power=200,
    samples=64
)
scene.add_light(main_light)

fill_light = AreaLight(
    position=[-2, 2, -1],
    normal=[0, -1, 0],
    width=1.0,
    height=1.0,
    power=80,
    samples=64
)
scene.add_light(fill_light)

# =========================
# CONSTROI BVH
# =========================
scene.build_bvh()

# =========================
# CAMERA
# =========================
camera = Camera(
    eye=[0, 1.5, 2],
    center=[0, 0.5, -3],
    up=[0, 1, 0],
    fov=np.pi / 3,
    aspect=1.0
)

# =========================
# FILME
# =========================
film = Film(
    width=300,
    height=300,
    samples=256
)

print("=" * 60)
print("PATH TRACING RENDERER")
print("=" * 60)
print(f"Resolution: {film.width}x{film.height}")
print(f"Samples per pixel: {film.samples}")
print(f"Total pixels: {film.width * film.height}")
print(f"Total rays: {film.width * film.height * film.samples}")
print("=" * 60)

# =========================
# RENDERING
# =========================
def render_row(j):
    row = np.zeros((film.width, 3), dtype=np.float32)
    for i in range(film.width):
        color = np.zeros(3)
        for _ in range(film.sample_count()):
            x, y = film.get_sample(i, j)
            ray = camera.generate_ray(x, y)
            color += scene.trace_ray(ray)

        color /= film.sample_count()
        color *= 1.25
        color = np.maximum(color, 0.0)
        color = np.power(color, 1.0 / 2.2)
        color = np.clip(color, 0.0, 1.0)
        row[i] = color

    return j, row

if __name__ == "__main__":
    workers = max(1, mp.cpu_count() - 1)

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for count, (j, row) in enumerate(
            executor.map(render_row, range(film.height)), 1
        ):

            if count % 10 == 0 or count == film.height:
                print(
                    f"Rendering line {count}/{film.height} "
                    f"({100*count/film.height:.1f}%)"
                )

            for i in range(film.width):
                film.set_pixel(i, j, row[i])

    print("=" * 60)
    print("Renderização concluída! Salvando imagem...")
    print("=" * 60)

    film.save("render_pathtracing.png")

    print("Imagem salva como: render_pathtracing.png")
    print("=" * 60)