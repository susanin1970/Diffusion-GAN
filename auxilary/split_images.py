from pathlib import Path
from PIL import Image
import argparse
import sys

DEFAULT_EXTS = {"png", "jpg", "jpeg", "tif", "tiff", "webp"}


def split_image(
    image_path: Path, out_dir: Path, tile_w=1024, tile_h=1024, cols=7, rows=4
):
    """Разбивает image_path на cols x rows тайлов размером tile_w x tile_h и сохраняет в out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        with Image.open(image_path) as im:
            w, h = im.size
            expected_w = cols * tile_w
            expected_h = rows * tile_h
            if (w, h) != (expected_w, expected_h):
                print(
                    f"Warning: {image_path} expected {(expected_w, expected_h)}, got {(w,h)}",
                    file=sys.stderr,
                )
            for row in range(rows):
                for col in range(cols):
                    left = col * tile_w
                    upper = row * tile_h
                    right = left + tile_w
                    lower = upper + tile_h
                    box = (left, upper, min(right, w), min(lower, h))
                    tile = im.crop(box)
                    if tile.size != (tile_w, tile_h):
                        canvas = Image.new(im.mode, (tile_w, tile_h))
                        canvas.paste(tile, (0, 0))
                        tile = canvas
                    out_path = (
                        out_dir / f"{image_path.stem}_r{row}_c{col}{image_path.suffix}"
                    )
                    tile.save(out_path)
        return True
    except Exception as e:
        print(f"Error processing {image_path}: {e}", file=sys.stderr)
        return False


def process_directory(
    input_dir: Path, out_root: Path, exts, recursive: bool, tile_w, tile_h, cols, rows
):
    """Проходит по директории и применяет split_image ко всем подходящим файлам."""
    input_dir = input_dir.resolve()
    out_root = out_root.resolve()
    files_processed = 0
    files_failed = 0

    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    for p in iterator:
        if not p.is_file():
            continue
        if p.suffix.lower().lstrip(".") not in exts:
            continue
        # Вычисляем целевую подпапку. При рекурсивном режиме сохраняем структуру.
        if recursive:
            try:
                rel_parent = p.parent.relative_to(input_dir)
                out_dir = out_root / rel_parent
            except Exception:
                out_dir = out_root
        else:
            out_dir = out_root
        success = split_image(
            p, out_dir, tile_w=tile_w, tile_h=tile_h, cols=cols, rows=rows
        )
        if success:
            files_processed += 1
        else:
            files_failed += 1

    print(
        f"Done. Processed: {files_processed}, Failed: {files_failed}, Output: {out_root}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split image file or all images in a directory into 7x4 1024x1024 tiles"
    )
    parser.add_argument(
        "input", help="Path to an image file or a directory with images"
    )
    parser.add_argument(
        "-o", "--out", default="tiles", help="Output root directory (default: tiles)"
    )
    parser.add_argument(
        "--exts",
        default=",".join(sorted(DEFAULT_EXTS)),
        help="Comma-separated list of extensions to process (default: png,jpg,jpeg,tif,tiff,webp)",
    )
    parser.add_argument(
        "--recursive", action="store_true", help="Recursively process subdirectories"
    )
    parser.add_argument(
        "--tile-w", type=int, default=1024, help="Tile width (default: 1024)"
    )
    parser.add_argument(
        "--tile-h", type=int, default=1024, help="Tile height (default: 1024)"
    )
    parser.add_argument(
        "--cols", type=int, default=7, help="Number of columns (default: 7)"
    )
    parser.add_argument(
        "--rows", type=int, default=4, help="Number of rows (default: 4)"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_root = Path(args.out)
    exts = {e.strip().lower() for e in args.exts.split(",") if e.strip()}

    if input_path.is_file():
        # Один файл
        split_image(
            input_path,
            out_root,
            tile_w=args.tile_w,
            tile_h=args.tile_h,
            cols=args.cols,
            rows=args.rows,
        )
    elif input_path.is_dir():
        process_directory(
            input_path,
            out_root,
            exts,
            args.recursive,
            tile_w=args.tile_w,
            tile_h=args.tile_h,
            cols=args.cols,
            rows=args.rows,
        )
    else:
        print(f"Input path does not exist: {input_path}", file=sys.stderr)
        sys.exit(1)
