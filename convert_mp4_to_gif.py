import imageio

def convert_mp4_to_gif(input_path, output_path):
    print(f"Converting {input_path} to {output_path}...")
    try:
        reader = imageio.get_reader(input_path)
        fps = reader.get_meta_data().get('fps', 10)
        
        # Write to GIF. We can limit fps or optimize it if it gets too large.
        writer = imageio.get_writer(output_path, fps=fps, quantizer='nq')
        
        for i, frame in enumerate(reader):
            # Skip frames to reduce file size (e.g. effective 10 fps) if original is high
            writer.append_data(frame)
            
        writer.close()
        print("Success!")
    except Exception as e:
        print(f"Error during conversion: {e}")

if __name__ == '__main__':
    convert_mp4_to_gif("assets/demo_video.mp4", "assets/demo_video.gif")
