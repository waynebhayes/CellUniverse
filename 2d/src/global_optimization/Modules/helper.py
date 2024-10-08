from PIL import Image
import numpy as np
from pathlib import Path

def get_input_file_paths(args, num_z_slices):
    """Gets the list of images that are to be analyzed."""

    # temporarily divide z_slices by 10
    input_file_paths = []

    if args.frame_first > args.frame_last and args.frame_last >= 0:
        raise ValueError('Invalid interval: frame_first must be less than frame_last')
    elif args.frame_first < 0:
        raise ValueError('Invalid interval: frame_first must be greater or equal to 0')

    i = args.frame_first
    while i == -1 or i <= args.frame_last:
        input_file_stack = []
        for z in range(num_z_slices):
            if num_z_slices == 1:
                file = Path(args.input % i)
            else:
                file = Path(args.input % (i, z))
            if file.exists() and file.is_file():
                input_file_stack.append((file, i, z))
            else:
                raise ValueError(f'Input file not found "{file}"')
        i += 1
        input_file_paths.append(input_file_stack)

    if i != -1 and len(input_file_paths) != args.frame_last - args.frame_first + 1:
        raise ValueError(f'Input files missing from interval [{args.frame_first}, {args.frame_last})')

    return input_file_paths

def load_image(image_file_name):
    """Open the image file and convert to a floating-point grayscale array."""
    with open(image_file_name, 'rb') as fp:
        real_image = np.array(Image.open(fp))
    if real_image.dtype == np.uint8:
        real_image = real_image.astype(float) / 255
    if len(real_image.shape) == 3:
        real_image = np.mean(real_image, axis=-1)
    return real_image
