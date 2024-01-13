import os
from PIL import Image as PILImage

def process_images(input_folder, output_folder):

    # If there isn't output folder yet, it creates one
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)

    for idx, file_name in enumerate(files):
        # Checks the folder if the image is a JPG
        if file_name.lower().endswith('.jpg'):
            # Makes the path of files with combined folder
            input_path = os.path.join(input_folder, file_name)

            # Generating needed file number
            output_name = f"{1 + idx}.jpg"
            output_path = os.path.join(output_folder, output_name)

            # Image processing and saving
            image = PILImage.open(input_path)
            image = image.convert("RGB")
            image = image.resize((256, 256))
            image.save(output_path)

if __name__ == "__main__":
    input_folder = r"censored" # i simply done that part 3 times, one for head, one for tail, one for perpendicular throw
    output_folder = r"censored"

    process_images(input_folder, output_folder)