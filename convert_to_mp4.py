import imageio

def convert_webp_to_mp4(input_path, output_path):
    print(f"Converting {input_path} to {output_path}...")
    try:
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data().get('fps', 10)
        
        writer = imageio.get_writer(output_path, fps=fps)
        
        for frame in reader:
            writer.append_data(frame)
            
        writer.close()
        print("Success!")
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == '__main__':
    import sys
    convert_webp_to_mp4(sys.argv[1], sys.argv[2])
