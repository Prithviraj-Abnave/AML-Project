from PIL import Image
import sys

def convert_webp_to_gif(input_path, output_path):
    print(f"Converting {input_path} to {output_path}...")
    try:
        im = Image.open(input_path)
        im.save(output_path, 'gif', save_all=True, optimize=False)
        print("Success!")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    convert_webp_to_gif(input_file, output_file)
