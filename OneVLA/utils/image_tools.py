from PIL import Image


def resize_images(images, target_size=(224, 224)):
    """
    Recursively resize all images in the nested list.

    :param images: nested list of images or single image.
    :param target_size: target size (width, height) after resizing.
    :return: resized images list, keeping the original nested structure.
    """
    if isinstance(images, Image.Image):
        return images.resize(target_size)
    elif isinstance(images, list):
        return [resize_images(img, target_size) for img in images]
    else:
        raise ValueError("Unsupported image type or structure.")
