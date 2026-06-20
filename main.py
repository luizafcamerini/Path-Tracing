import numpy as np
from components.camera import Camera
from components.material import DiffuseMaterial, PhongMaterial
from components.scene import Scene
from components.shapes import Sphere, Plane, Box
from components.light import PointLight, AreaLight
from components.film import Film
from components.transform import Transform
from components.instance import Instance

scene = Scene()

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
    gray_material,
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
    green_material,
    transform=Transform(
        translation=[0, 2, -5]
    )
)
scene.add_object(back_plane)

# =========================
# PLANO LATERAL ESQUERDO
# =========================
left_plane = Instance(
    Plane(
        normal=[1, 0, 0],
        material=DiffuseMaterial(albedo=[0.5, 0.1, 0.1])
    ),
    DiffuseMaterial(albedo=[0.5, 0.1, 0.1]),
    transform=Transform(
        translation=[-3, 2, -3]
    )
)
scene.add_object(left_plane)

# =========================
# PLANO LATERAL DIREITO
# =========================
right_plane = Instance(
    Plane(
        normal=[-1, 0, 0],
        material=DiffuseMaterial(albedo=[0.1, 0.1, 0.5])
    ),
    DiffuseMaterial(albedo=[0.1, 0.1, 0.5]),
    transform=Transform(
        translation=[3, 2, -3]
    )
)
scene.add_object(right_plane)

# =========================
# LUZES RETANGULARES
# =========================
main_light = AreaLight(
    position=[0, 3.5, -3],
    normal=[0, -1, 0],
    width=2.0,
    height=2.0,
    power=60,
    samples=1
)
scene.add_light(main_light)

fill_light = AreaLight(
    position=[-2, 2, -1],
    normal=[0, -1, 0],
    width=1.0,
    height=1.0,
    power=25,
    samples=1
)
scene.add_light(fill_light)

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
    width=400,
    height=400,
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
for j in range(film.height):
    print(f"Rendering line {j+1}/{film.height} ({100*(j+1)/film.height:.1f}%)")
    for i in range(film.width):
        color = np.array([0.0, 0.0, 0.0])
        
        for sample_idx in range(film.sample_count()):
            x, y = film.get_sample(i, j)
            ray = camera.generate_ray(x, y)
            
            sample_color = scene.trace_ray(ray, depth=0)
            color += sample_color
        
        color /= film.sample_count()
        
        # Tone mapping com exposure control
        exposure = 1.0  # Ajuste: < 1.0 para escurecer, > 1.0 para clarear
        color = color * exposure
        
        # Clipping e gamma correction
        color = np.power(np.clip(color, 0, 1), 1.0 / 2.2)
        
        film.set_pixel(i, j, color)

print("=" * 60)
print("Renderização concluída! Salvando imagem...")
print("=" * 60)

film.save("render_pathtracing.png")

print("Imagem salva como: render_pathtracing.png")
print("=" * 60)