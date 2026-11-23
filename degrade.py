#!/usr/bin/env python3
"""
degrade.py — делает из "хорошей" фотографии фото в стиле дешёвой/старой камеры
(референс: фронталка, тёмное помещение, шум, мыло, JPEG-артефакты, холодный оттенок).

Использование:
    python3 degrade.py photo.jpg                 # -> photo_degraded.jpg рядом
    python3 degrade.py photo1.jpg photo2.png     # несколько файлов
    python3 degrade.py ./input_folder -o ./out   # целая папка (пачка)
    python3 degrade.py ./in -o ./out --strength 1.3 --seed 42

Зависимости: pip install pillow numpy
"""
import argparse
import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic"}

# ---------- параметры "профиля" (подогнаны под референс-фото) ----------
PROFILE = dict(
    long_side=1280,        # размер длинной стороны (референс 960x1280)
    blur=0.7,              # радиус мыла после даунскейла (px)
    black_point=14.0,      # подъём чёрного -> провал в тенях (0..255 шкала)
    gamma=1.06,            # >1 темнее полутона
    contrast=1.08,         # S-кривая
    exposure=1.0,         # общая яркость
    gain_rgb=(0.93, 0.99, 1.07),   # холодный баланс белого (R,G,B)
    shadow_tint=(-2.0, 0.0, 5.0),  # синева в тенях
    saturation=1.02,
    luma_noise=4.5,        # сигма яркостного шума (на шкале 0..255)
    chroma_noise=2.0,      # сигма цветового шума
    noise_shadow_boost=1.8,  # шум сильнее в тенях
    noise_grain=0.7,       # радиус сглаживания шума (зерно, а не "соль")
    vignette=0.18,
    jpeg_quality=72,
    jpeg_passes=2,         # двойное пересжатие (как мессенджер)
    jpeg_final_quality=78,
)


def to_float(img):
    return np.asarray(img.convert("RGB"), dtype=np.float32)


def to_img(a):
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGB")


def blur_noise(n, radius):
    """Сделать шум мелким зерном, а не пиксельной крупой."""
    if radius <= 0:
        return n
    r = max(1, int(radius * 3))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / radius) ** 2).astype(np.float32)
    k /= k.sum()
    pad = np.pad(n, ((r, r), (r, r), (0, 0)), mode="reflect")
    out = sum(k[i] * pad[i:i + n.shape[0]] for i in range(2 * r + 1))
    out = sum(k[i] * out[:, i:i + n.shape[1]] for i in range(2 * r + 1))
    return out * (n.std() / (out.std() + 1e-6))  # вернуть амплитуду шума


def degrade(img, rng, s=1.0, p=PROFILE):
    """s — общий множитель силы (1.0 = как референс)."""
    img = ImageOps.exif_transpose(img).convert("RGB")  # учесть поворот из EXIF

    # 1. даунскейл до "дешёвой матрицы"
    w, h = img.size
    scale = p["long_side"] / max(w, h)
    if scale < 1:
        # чуть вариативный фильтр для лёгких артефактов ресайза
        img = img.resize((round(w * scale), round(h * scale)), Image.BILINEAR
                         if scale < 0.5 else Image.LANCZOS)
    # 2. мыло оптики/шумодава
    if p["blur"] * s > 0:
        img = img.filter(ImageFilter.GaussianBlur(p["blur"] * s))

    a = to_float(img) / 255.0

    # 3. экспозиция / тон
    a = a * (p["exposure"] ** s)
    bp = p["black_point"] * s / 255.0
    a = np.clip((a - bp) / (1 - bp), 0, 1)
    a = a ** (1 + (p["gamma"] - 1) * s)
    c = 1 + (p["contrast"] - 1) * s
    a = np.clip(0.5 + (a - 0.5) * c, 0, 1)

    # 4. цвет: холодный WB + синева в тенях + насыщенность
    gain = np.array(p["gain_rgb"], dtype=np.float32)
    a = a * (1 + (gain - 1) * s)
    lum = a.mean(2, keepdims=True)
    shadow = (1 - lum) ** 2
    a = a + shadow * (np.array(p["shadow_tint"], dtype=np.float32) / 255.0) * s
    lum = a.mean(2, keepdims=True)
    a = lum + (a - lum) * (1 + (p["saturation"] - 1) * s)
    a = np.clip(a, 0, 1) * 255.0

    # 5. виньетка
    hh, ww = a.shape[:2]
    yy, xx = np.mgrid[0:hh, 0:ww].astype(np.float32)
    r = np.sqrt(((xx - ww / 2) / (ww / 2)) ** 2 + ((yy - hh / 2) / (hh / 2)) ** 2)
    a *= (1 - p["vignette"] * s * np.clip(r - 0.4, 0, None) ** 2)[..., None]

    # 6. шум: яркостной + цветовой, сильнее в тенях
    lum = a.mean(2, keepdims=True) / 255.0
    amp = 1 + p["noise_shadow_boost"] * (1 - lum) ** 2
    ln = rng.standard_normal((hh, ww, 1)).astype(np.float32)
    cn = rng.standard_normal((hh, ww, 3)).astype(np.float32)
    ln = blur_noise(np.repeat(ln, 3, 2), p["noise_grain"])[..., :1]
    cn = blur_noise(cn, p["noise_grain"] * 1.6)
    cn -= cn.mean(2, keepdims=True)
    a += (ln * p["luma_noise"] + cn * p["chroma_noise"]) * amp * s

    img = to_img(a)

    # 7. JPEG: 4:2:0, пересжатие
    q = int(np.clip(100 - (100 - p["jpeg_quality"]) * s, 5, 95))
    for _ in range(max(1, p["jpeg_passes"] - 1)):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q, subsampling=2)
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img, int(np.clip(100 - (100 - p["jpeg_final_quality"]) * s, 5, 95))


def collect(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS)
        elif p.exists():
            files.append(p)
        else:
            print(f"! не найден: {p}", file=sys.stderr)
    return files


def main():
    ap = argparse.ArgumentParser(description="Деградация фото под дешёвую камеру")
    ap.add_argument("inputs", nargs="+", help="файлы и/или папки")
    ap.add_argument("-o", "--out", help="папка результата (по умолчанию рядом с исходником)")
    ap.add_argument("--strength", type=float, default=1.0, help="сила эффекта, 1.0 = референс")
    ap.add_argument("--long-side", type=int, help="длинная сторона результата (по умолч. 1280)")
    ap.add_argument("--seed", type=int, help="seed шума (воспроизводимость)")
    ap.add_argument("--mirror", action="store_true", help="отзеркалить как фронтальная камера")
    a = ap.parse_args()

    prof = dict(PROFILE)
    if a.long_side:
        prof["long_side"] = a.long_side
    out_dir = Path(a.out) if a.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    files = collect(a.inputs)
    if not files:
        sys.exit("нет входных файлов")
    for i, f in enumerate(files):
        rng = np.random.default_rng(None if a.seed is None else a.seed + i)
        with Image.open(f) as im:
            if a.mirror:
                im = ImageOps.mirror(ImageOps.exif_transpose(im))
            res, fq = degrade(im, rng, a.strength, prof)
        dst = (out_dir or f.parent) / f"{f.stem}_degraded.jpg"
        res.save(dst, "JPEG", quality=fq, subsampling=2)  # без EXIF
        print(f"[{i+1}/{len(files)}] {f.name} -> {dst} {res.size}")


if __name__ == "__main__":
    main()
