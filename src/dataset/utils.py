import os
import colorsys
import cairosvg

def get_hsv_from_z(z_hsv):
    """Returns a tuple of (h, s, v) and the hex string for internal use.
    Input z_hsv should be an array/tuple of length 3."""
    h = z_hsv[0]
    s = 0.5 + z_hsv[1] * 0.5 # Vibrant range
    v = 0.5 + z_hsv[2] * 0.5 # Vibrant range
    
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    hex_val = '#{:02x}{:02x}{:02x}'.format(int(r*255), int(g*255), int(b*255))
    
    return {"hsv": [h, s, v], "rgb": [r, g, b], "hex": hex_val}

def make_mouth_path(cx, cy, width, curve):
    """Creates a Quadratic Bezier path for the mouth based on curve value."""
    x1 = cx - width / 2
    x2 = cx + width / 2
    control_y = cy + curve
    return f"M {x1:.2f} {cy:.2f} Q {cx:.2f} {control_y:.2f} {x2:.2f} {cy:.2f}"

def convert_svgs_to_pngs(svg_dir, png_dir, canvas_w=224, canvas_h=224):
    print(f"Starting conversion: SVG -> PNG ({png_dir})")
    os.makedirs(png_dir, exist_ok=True)

    count = 0
    for filename in os.listdir(svg_dir):
        if filename.endswith(".svg"):
            svg_path = os.path.join(svg_dir, filename)
            png_name = filename.replace(".svg", ".png")
            png_path = os.path.join(png_dir, png_name)

            # Skip conversion if PNG already exists to allow resuming
            if not os.path.exists(png_path):
                try:
                    cairosvg.svg2png(
                        url=svg_path,
                        write_to=png_path,
                        output_width=canvas_w,
                        output_height=canvas_h
                    )
                    count += 1
                    if count % 1000 == 0:
                        print(f"Converted {count} new images...")
                except Exception as e:
                    print(f"WARNING: Skipping corrupted SVG file {filename}")

    print(f"Successfully converted remaining files! New converted: {count}")
