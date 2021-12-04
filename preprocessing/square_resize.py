import numpy as np
from skimage.transform import resize

MAP_INTERPOLATION_TO_ORDER = {
    "nearest": 0,
    "bilinear": 1,
    "biquadratic": 2,
    "bicubic": 3,
}


def center_crop_and_resize(image, image_size, crop_padding=32, interpolation="bicubic"):
    assert image.ndim in {2, 3}
    assert interpolation in MAP_INTERPOLATION_TO_ORDER.keys()

    h, w = image.shape[:2]

    padded_center_crop_size = int(
        (image_size / (image_size + crop_padding)) * min(h, w)
    )
    offset_height = ((h - padded_center_crop_size) + 1) // 2
    offset_width = ((w - padded_center_crop_size) + 1) // 2

    image_crop = image[
                 offset_height: padded_center_crop_size + offset_height,
                 offset_width: padded_center_crop_size + offset_width,
                 ]
    resized_image = resize(
        image_crop,
        (image_size, image_size),
        order=MAP_INTERPOLATION_TO_ORDER[interpolation],
        preserve_range=True,
    )

    return resized_image


def square_resize(img):
    max_length = max(img.shape)
    
    old_size = img.shape[:2] # old_size is in (height, width) format

    ratio = float(max_length)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = max_length - new_size[1]
    delta_h = max_length - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    ret_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
        value=color)
    
    #ret_img = tf.image.resize_with_pad(img, max_length, max_length, method=ResizeMethod.BILINEAR, antialias=False)
    return ret_img

def skimage_resize(image, w=299, h=299, interpolation="bicubic"):
    assert interpolation in MAP_INTERPOLATION_TO_ORDER.keys()
    resized_image = resize(image, (h, w),
        order=MAP_INTERPOLATION_TO_ORDER[interpolation],
        preserve_range=True)
    return resized_image
