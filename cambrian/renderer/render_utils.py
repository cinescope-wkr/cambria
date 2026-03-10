"""Rendering utilities."""

from typing import Dict

import mujoco as mj
import torch
import torch.nn.functional as F


def resize(images: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Resize the image to the specified height and width."""
    squeeze = False
    if images.ndim == 3:
        squeeze = True
        images = images.unsqueeze(0)

    resized_images = F.interpolate(
        images.permute(0, 3, 1, 2),
        size=(height, width),
        mode="nearest",
        align_corners=None,
    ).permute(0, 2, 3, 1)

    return resized_images.squeeze(0) if squeeze else resized_images


def resize_with_aspect_fill(
    images: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Resize the image while maintaining the aspect ratio and
    filling the rest with black."""
    squeeze = False
    if images.ndim == 3:
        squeeze = True
        images = images.unsqueeze(0)

    original_height, original_width = images.shape[1:3]
    ratio_original = original_width / original_height
    ratio_new = width / height

    transpose = False
    if ratio_original < ratio_new:
        # Transpose for taller images
        transpose = True
        images = images.permute(0, 2, 1, 3)
        height, width = width, height
        ratio_new = width / height
        ratio_original = original_height / original_width

    resize_height = max(1, round(width / ratio_original))
    resized_images = resize(images, resize_height, width)

    pad_top = (height - resize_height) // 2
    pad_bottom = height - resize_height - pad_top
    padded_images = F.pad(
        resized_images.permute(0, 3, 1, 2),
        (0, 0, pad_top, pad_bottom),
        mode="constant",
        value=0,
    ).permute(0, 2, 3, 1)

    if transpose:
        # Transpose back
        padded_images = padded_images.permute(0, 2, 1, 3)

    return padded_images.squeeze(0) if squeeze else padded_images


def add_border(
    images: torch.Tensor,
    border_size: int,
    color: tuple = (0, 0, 0),
) -> torch.Tensor:
    squeeze = False
    if images.ndim == 3:
        squeeze = True
        images = images.unsqueeze(0)
    color = torch.tensor(color, device=images.device).unsqueeze(-1).unsqueeze(-1)
    pad = (border_size, border_size, border_size, border_size)
    images = images.permute(0, 3, 1, 2)
    padded = F.pad(images, pad, value=0.0)
    padded[..., :border_size, :] = color
    padded[..., -border_size:, :] = color
    padded[..., :, :border_size] = color
    padded[..., :, -border_size:] = color
    return padded.squeeze(0).permute(1, 2, 0) if squeeze else padded.permute(0, 2, 3, 1)


def generate_composite(images: Dict[float, Dict[float, torch.Tensor]]) -> torch.Tensor:
    """This is a debug method which renders the images as a composite image.

    Will appear as a compound eye. For example, if we have a 3x3 grid of eyes:
        TL T TR
        ML M MR
        BL B BR

    Each eye has a red border around it.

    Note:

        This assumes that the images have the same dimensions.
    """
    composite = torch.stack(
        [
            images[lat][lon]
            for lat in sorted(images.keys())
            for lon in sorted(images[lat].keys())[::-1]
        ]
    )
    _, H, W, _ = composite.shape
    if H > W:
        h = max(H, 10)
        w = int(W * h / H)
    else:
        w = max(W, 10)
        h = int(H * w / W)
    composite = resize_with_aspect_fill(composite, h, w)
    composite = add_border(composite, 1, color=(1, 0, 0))

    # Resize the composite image while maintaining the spatial position
    _, H, W, C = composite.shape
    nrows, ncols = len(images), len(next(iter(images.values())))
    composite = (
        composite.view(nrows, ncols, H, W, C)
        .permute(0, 2, 1, 3, 4)
        .reshape(nrows * H, ncols * W, C)
    )

    return composite


def convert_depth_distances(model: mj.MjModel, depth: torch.Tensor) -> torch.Tensor:
    """Converts depth values from OpenGL to metric depth values using PyTorch.

    Args:
        model (mj.MjModel): The model.
        depth (torch.Tensor): The depth values to convert.

    Returns:
        torch.Tensor: The converted depth values.

    Note:
        This function is based on
        [this code](https://github.com/google-deepmind/mujoco/blob/main/\
            python/mujoco/renderer.py).
        It is adapted to use PyTorch instead of NumPy.
    """

    # Get the distances to the near and far clipping planes.
    extent = model.stat.extent
    near = model.vis.map.znear * extent
    far = model.vis.map.zfar * extent

    # Calculate OpenGL perspective matrix values in float32 precision
    # so they are close to what glFrustum returns
    # https://registry.khronos.org/OpenGL-Refpages/gl2.1/xhtml/glFrustum.xml
    zfar = torch.tensor(far, dtype=torch.float32)
    znear = torch.tensor(near, dtype=torch.float32)
    c_coef = -(zfar + znear) / (zfar - znear)
    d_coef = -(torch.tensor(2.0, dtype=torch.float32) * zfar * znear) / (zfar - znear)

    # In reverse Z mode the perspective matrix is transformed by the following
    c_coef = torch.tensor(-0.5, dtype=torch.float32) * c_coef - torch.tensor(
        0.5, dtype=torch.float32
    )
    d_coef = torch.tensor(-0.5, dtype=torch.float32) * d_coef

    # We need 64 bits to convert Z from ndc to metric depth without noticeable
    # losses in precision
    out_64 = depth.to(dtype=torch.float64)

    # Undo OpenGL projection
    # Note: We do not need to take action to convert from window coordinates
    # to normalized device coordinates because in reversed Z mode the mapping
    # is identity
    out_64 = d_coef / (out_64 + c_coef)

    # Cast result back to float32 for backwards compatibility
    # This has a small accuracy cost
    return out_64.to(dtype=torch.float32)


def convert_depth_to_rgb(
    depth: torch.Tensor, znear: float | None = None, zfar: float | None = None
) -> torch.Tensor:
    """Converts depth values to RGB values.

    Args:
        model (mj.MjModel): The model.
        depth (torch.Tensor): The depth values.

    Returns:
        torch.Tensor: The RGB values.
    """
    znear = znear or depth.min()
    zfar = zfar or depth.max()
    if znear != zfar:
        depth = (depth - znear) / (zfar - znear)
        depth = 1 - torch.clamp(depth, 0.0, 1.0)
    depth = depth.repeat(3, 1, 1).permute(1, 2, 0)
    return depth


def add_text(
    image: torch.Tensor,
    text: str,
    position: tuple[int, int] = (0, 0),
    size: int | None = None,
    **kwargs,
) -> torch.Tensor:
    """Add text to an image.

    Note:
        This is slow, so use it sparingly.

    Args:
        image (torch.Tensor): The image to add text to.
        text (str): The text to add.
        position (tuple[int, int]): The position to add the text.
        color (tuple[int, int, int], optional): The color of the text.
            Defaults to (255, 255, 255).

    Returns:
        torch.Tensor: The image with the text added.
    """
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    device = image.device

    image = Image.fromarray((torch.flipud(image) * 255).cpu().numpy().astype("uint8"))
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size)
    draw.text(position, text, font=font, **kwargs)
    image = torch.tensor(np.array(image), device=device) / 255
    image = torch.flipud(image)

    return image
