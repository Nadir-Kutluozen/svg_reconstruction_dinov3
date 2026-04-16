import os
import json

# Import the generation logic from our other file
from generate_svg import generate_two_faces_svg, SVG_DIR, META_DIR

print("Testing the SVG generation...")

# Generate just a couple of examples
num_examples = 3

for i in range(num_examples):
    sample_id = f"test_face_{i}"
    
    # Run the imported function
    svg_content, meta_data = generate_two_faces_svg(sample_id)

    # Save SVG
    svg_path = os.path.join(SVG_DIR, f"{sample_id}.svg")
    with open(svg_path, "w", encoding="utf-8") as f:
        f.write(svg_content)

    # Save Metadata
    meta_path = os.path.join(META_DIR, f"{sample_id}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_data, f, indent=2)

print(f"Successfully generated {num_examples} examples!")
print(f"Check {SVG_DIR} and {META_DIR} to see them.")
