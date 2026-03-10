"""This is an optics-enabled eye, which implements a height map and a PSF on top
of the existing eye."""

from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Optional, Self, Tuple

import torch
from hydra_config import HydraContainerConfig, config_wrapper

from cambrian.eyes.eye import MjCambrianEye, MjCambrianEyeConfig
from cambrian.utils import device, make_odd


@config_wrapper
class MjCambrianApertureConfig(HydraContainerConfig, ABC):
    @abstractmethod
    def calculate_aperture_mask(
        self,
        X1: torch.Tensor,
        Y1: torch.Tensor,
        X1_Y1: torch.Tensor,
        Lx: float,
        Ly: float,
    ) -> torch.Tensor:
        """This method calculates the aperture mask.

        Args:
            X1 (torch.Tensor): Horizontal coordinate grid in meters.
            Y1 (torch.Tensor): Vertical coordinate grid in meters.
            X1_Y1 (torch.Tensor): Squared distance from the center of the aperture.
            Lx (float): Width of the aperture.
            Ly (float): Height of the aperture.

        Returns:
            torch.Tensor: Aperture mask.
        """
        pass


@config_wrapper
class MjCambrianCircularApertureConfig(MjCambrianApertureConfig):
    """This defines the config for the circular aperture. This extends the base aperture
    config and adds additional parameters for the circular aperture.

    Attributes:
        radius (float): Radius of the circular aperture.
    """

    radius: float

    def calculate_aperture_mask(
        self,
        X1: torch.Tensor,
        Y1: torch.Tensor,
        X1_Y1: torch.Tensor,
        Lx: float,
        Ly: float,
    ) -> torch.Tensor:
        aperture_radius = min(Lx / 2, Ly / 2) * self.radius + 1e-7
        return torch.nan_to_num(torch.sqrt(X1_Y1) / aperture_radius) <= 1.0


@config_wrapper
class MjCambrianEllipticalApertureConfig(MjCambrianApertureConfig):
    """This defines the config for an elliptical aperture.

    Attributes:
        radius_x (float): Relative radius along X.
        radius_y (float): Relative radius along Y.
    """

    radius_x: float
    radius_y: float

    def calculate_aperture_mask(
        self,
        X1: torch.Tensor,
        Y1: torch.Tensor,
        X1_Y1: torch.Tensor,
        Lx: float,
        Ly: float,
    ) -> torch.Tensor:
        rx = (Lx / 2.0) * self.radius_x + 1e-7
        ry = (Ly / 2.0) * self.radius_y + 1e-7
        mask = (X1 / rx).square() + (Y1 / ry).square()
        return torch.nan_to_num(mask) <= 1.0


@config_wrapper
class MjCambrianMaskApertureConfig(MjCambrianApertureConfig):
    """This defines the config for the custom aperture. This extends the base aperture
    config and adds additional parameters for the custom aperture.

    Attributes:
        mask (Optional[List[List[int]]]): Aperture mask. This is a 2D array that defines
            the aperture mask. The aperture mask is a binary mask that defines the
            aperture of the lens. It's a binary mask where 1 lets light through and 0
            blocks it. The mask can only be None if randomize is True or if size is
            not None. Defaults to None.
        randomize (bool): Randomize the aperture mask. If True, the aperture mask is
            randomized.
        random_prob (Optional[float]): Probability of the aperture mask being 1. If
            None, the probability is 0.5. Defaults to None.
        size (Optional[Tuple[int, int]]): Size of the aperture mask. If None, the size
            is the same as the pupil resolution. Defaults to None.
    """

    mask: Optional[List[List[int]]] = None
    randomize: bool
    random_prob: Optional[float] = None
    size: Optional[Tuple[int, int]] = None

    def calculate_aperture_mask(self, X1: torch.Tensor, Y1: torch.Tensor, X1_Y1: torch.Tensor, *_) -> torch.Tensor:
        size = self.size if self.size is not None else X1_Y1.shape
        if self.mask is None:
            assert self.randomize or self.size is not None, "Mask or size must be set."
            mask = torch.randint(0, 2, size, dtype=torch.float32)
            if self.random_prob is not None:
                mask = mask > self.random_prob
        else:
            mask = torch.tensor(self.mask, dtype=torch.float32)
        assert mask.shape[0] == mask.shape[1]
        if self.randomize:
            mask = torch.randint(0, 2, mask.shape, dtype=torch.float32)

        mask = (
            torch.nn.functional.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                size=size,
                mode="bicubic",
            )
            .squeeze(0)
            .squeeze(0)
        )
        return mask > 0.5


@config_wrapper
class MjCambrianOpticsEyeConfig(MjCambrianEyeConfig):
    """This defines the config for the optics module. This extends the base eye config
    and adds additional parameters for the optics module.

    Attributes:
        pupil_resolution (Tuple[int, int]): Resolution of the pupil plane. This
            is used to calculate the PSF.

        noise_std (float): Standard deviation of the Gaussian noise to be
            added to the image. If 0.0, no noise is added.
        wavelengths (Tuple[float, float, float]): Wavelengths of the RGB channels.

        f_stop (float): F-stop of the lens. This is used to calculate the PSF.
        refractive_index (float): Refractive index of the lens material.
        height_map (List[float]): Legacy radial height profile of the lens. This is
            normalized to [0, 1] and treated as a radially symmetric profile.
        max_height_um (float): Physical maximum sag height of the lens in microns.
            This decouples lens geometry from refractive index.
        height_profile_ctrl (List[float]): Radial control points used to construct a
            smooth lens profile from center to edge.
        height_profile_poly (List[float]): Even polynomial coefficients applied as
            `1 - c2 r^2 - c4 r^4 - ...` over normalized radius.
        height_profile_mix (float): Blend between the polynomial profile and the
            control-point profile. `0` uses only the polynomial profile, `1` uses only
            the control-point profile.
        scale_intensity (bool): Whether to scale the intensity of the PSF by the
            overall throughput of the aperture.

        refractive_index_edge (Optional[float]): Edge refractive index used to build a
            quadratic GRIN profile when no explicit control profile is supplied.
        grin_profile_ctrl (List[float]): Radial refractive-index control points used to
            define a GRIN lens from center to edge.

        aperture (MjCambrianApertureConfig): Aperture config. This defines the
            aperture of the lens. The aperture can be circular or custom.

        depths (List[float]): Depths at which the PSF is calculated. If empty, the psf
            is calculated for each render call; otherwise, the PSFs are precomputed.
    """

    instance: Callable[[Self, str], "MjCambrianOpticsEye"]

    pupil_resolution: Tuple[int, int]

    noise_std: float
    wavelengths: Tuple[float, float, float]

    f_stop: float
    sensor_distance: Optional[float] = None
    refractive_index: float
    height_map: List[float]
    max_height_um: float
    height_profile_ctrl: List[float]
    height_profile_poly: List[float]
    height_profile_mix: float
    height_profile_ctrl_0: Optional[float] = None
    height_profile_ctrl_1: Optional[float] = None
    height_profile_ctrl_2: Optional[float] = None
    height_profile_ctrl_3: Optional[float] = None
    height_profile_poly_c2: Optional[float] = None
    height_profile_poly_c4: Optional[float] = None
    height_profile_poly_c6: Optional[float] = None
    scale_intensity: bool
    refractive_index_edge: Optional[float] = None
    grin_profile_ctrl: Optional[List[float]] = None
    grin_profile_ctrl_0: Optional[float] = None
    grin_profile_ctrl_1: Optional[float] = None
    grin_profile_ctrl_2: Optional[float] = None
    grin_profile_ctrl_3: Optional[float] = None

    aperture: MjCambrianApertureConfig

    depths: List[float]


class MjCambrianOpticsEye(MjCambrianEye):
    """This class applies the depth invariant PSF to the image.

    Args:
        config (MjCambrianOpticsConfig): Config for the optics module.
    """

    def __init__(self, config: MjCambrianOpticsEyeConfig, name: str):
        super().__init__(config, name)
        self._config: MjCambrianOpticsEyeConfig

        self._renders_depth = "depth_array" in self._config.renderer.render_modes
        assert self._renders_depth, "Eye: 'depth_array' must be a render mode."

        self._psfs: Dict[torch.Tensor, torch.Tensor] = {}
        self._depths = torch.tensor(self._config.depths).to(device)
        self.initialize()

    def initialize(self):
        """This will initialize the parameters used during the PSF calculation."""
        # pupil_Mx,pupil_My defines the number of pixels in x,y direction
        # (i.e. width, height) of the pupil
        pupil_Mx, pupil_My = torch.tensor(self._config.pupil_resolution)
        assert (
            pupil_Mx > 2 and pupil_My > 2
        ), f"Pupil resolution must be > 2: {pupil_Mx=}, {pupil_My=}"
        assert (
            pupil_Mx % 2 and pupil_My % 2
        ), f"Pupil resolution must be odd: {pupil_Mx=}, {pupil_My=}"

        # pupil_dx/pupil_dy defines the pixel pitch (m) (i.e. distance between the
        # centers of adjacent pixels) of the pupil and Lx/Ly defines the size of the
        # pupil plane
        fx, fy = self._config.focal
        Lx, Ly = fx / self._config.f_stop, fy / self._config.f_stop
        pupil_dx, pupil_dy = Lx / pupil_Mx, Ly / pupil_My

        # Image plane coords
        # TODO: fragile to floating point errors, must use double here. okay to convert
        # to float after psf operations
        x1 = torch.linspace(-Lx / 2.0, Lx / 2.0, pupil_Mx).double()
        y1 = torch.linspace(-Ly / 2.0, Ly / 2.0, pupil_My).double()
        X1, Y1 = torch.meshgrid(x1, y1, indexing="ij")
        X1_Y1 = X1.square() + Y1.square()

        # Frequency coords
        freqx = torch.linspace(
            -1.0 / (2.0 * pupil_dx), 1.0 / (2.0 * pupil_dx), pupil_Mx
        )
        freqy = torch.linspace(
            -1.0 / (2.0 * pupil_dy), 1.0 / (2.0 * pupil_dy), pupil_My
        )
        FX, FY = torch.meshgrid(freqx, freqy, indexing="ij")

        # Aperture mask
        A = self._config.aperture.calculate_aperture_mask(X1, Y1, X1_Y1, Lx, Ly)

        # Going to scale the intensity by the overall throughput of the aperture
        self._scaling_intensity = (A.sum() / (max(pupil_Mx * pupil_My, 1))) ** 2

        # Calculate the wave number
        wavelengths = torch.tensor(self._config.wavelengths).reshape(-1, 1, 1)
        k = 1j * 2 * torch.pi / wavelengths

        # Calculate the pupil from the height map
        # NOTE: Have to convert to numpy then to tensor to avoid issues with
        # MjCambrianConfigContainer
        maxr = ((Lx / 2.0) ** 2 + (Ly / 2.0) ** 2) ** 0.5
        maxr_px = float(torch.sqrt((pupil_Mx / 2).square() + (pupil_My / 2).square()).item())
        radial_bins = max(int(torch.ceil(torch.tensor(maxr_px)).item()), 2)
        h_r = self._build_height_profile(radial_bins)
        n_r = self._build_refractive_index_profile(radial_bins)
        r = torch.sqrt(X1_Y1)
        r_norm = torch.clamp(r / maxr, min=0.0, max=1.0)
        r_idx = torch.floor(r_norm * (radial_bins - 1)).to(torch.int64)
        height_map: torch.Tensor = h_r[r_idx] * (self._config.max_height_um * 1e-6)
        refractive_index_map: torch.Tensor = n_r[r_idx]
        phi_m = k * (refractive_index_map - 1.0) * height_map
        pupil = A * torch.exp(phi_m)

        # Determine the scaled down psf size. Will resample the psf such that the conv
        # is faster
        sx, sy = self._config.sensorsize
        scene_dx = sx / self._config.renderer.width
        scene_dy = sy / self._config.renderer.height
        psf_resolution = (make_odd(Lx / scene_dx), make_odd(Ly / scene_dy))

        # Pre-compute some values that are reused in the PSF calculation
        H_valid = torch.sqrt(FX.square() + FY.square()) < (1.0 / wavelengths)
        # NOTE: keep backward-compatible behavior by defaulting to fx when sensor distance
        # is not configured explicitly.
        sensor_distance = (
            float(self._config.sensor_distance)
            if self._config.sensor_distance is not None
            else float(fx)
        )
        FX_FY = torch.exp(
            sensor_distance
            * k
            * torch.sqrt(1 - (wavelengths * FX).square() - (wavelengths * FY).square())
        )
        H = H_valid * FX_FY

        # Now store all as class attributes
        self._X1, self._Y1 = X1.to(device), Y1.to(device)
        self._X1_Y1 = X1_Y1.to(device)
        self._H_valid = H_valid.to(device)
        self._H = H.to(device)
        self._FX, self._FY = FX.to(device), FY.to(device)
        self._FX_FY = FX_FY.to(device)
        self._k = k.to(device)
        self._A = A.to(device)
        self._pupil = pupil.to(device)
        self._height_map = height_map.to(device)
        self._radial_height_profile = h_r.to(device)
        self._refractive_index_map = refractive_index_map.to(device)
        self._radial_refractive_index_profile = n_r.to(device)
        self._psf_resolution = psf_resolution

        # Precompute the PSFs, if necessary
        if self._config.depths:
            self._precompute_psfs()

    def _build_height_profile(self, radial_bins: int) -> torch.Tensor:
        height_profile_ctrl = self._resolve_height_profile_ctrl()
        height_profile_poly = self._resolve_height_profile_poly()
        base_profile = self._interpolate_profile(
            self._config.height_map,
            radial_bins,
            default_value=0.5,
        )
        control_profile = self._interpolate_profile(
            height_profile_ctrl,
            radial_bins,
            default_value=base_profile[0].item(),
        )
        r_norm = torch.linspace(0.0, 1.0, radial_bins, dtype=torch.float64)

        polynomial_profile = torch.ones_like(r_norm)
        for index, coefficient in enumerate(height_profile_poly, start=1):
            polynomial_profile -= coefficient * r_norm ** (2 * index)
        polynomial_profile = torch.clamp(polynomial_profile, 0.0, 1.0)

        mix = float(min(max(self._config.height_profile_mix, 0.0), 1.0))
        height_profile = torch.lerp(polynomial_profile, control_profile, mix)

        if not height_profile_ctrl and not any(abs(coefficient) > 0 for coefficient in height_profile_poly):
            height_profile = base_profile

        return torch.clamp(height_profile, 0.0, 1.0)

    def _build_refractive_index_profile(self, radial_bins: int) -> torch.Tensor:
        r_norm = torch.linspace(0.0, 1.0, radial_bins, dtype=torch.float64)
        edge_index = (
            self._config.refractive_index
            if self._config.refractive_index_edge is None
            else self._config.refractive_index_edge
        )
        base_profile = edge_index + (
            self._config.refractive_index - edge_index
        ) * (1.0 - r_norm.square())

        grin_profile_ctrl = self._resolve_grin_profile_ctrl()
        has_explicit_grin_profile = grin_profile_ctrl and any(
            abs(value - self._config.refractive_index) > 1e-9 for value in grin_profile_ctrl
        )
        if has_explicit_grin_profile:
            grin_profile = self._interpolate_profile(
                grin_profile_ctrl,
                radial_bins,
                default_value=self._config.refractive_index,
                clamp_min=1.0,
                clamp_max=None,
            )
            return torch.clamp(grin_profile, min=1.0)

        return torch.clamp(base_profile, min=1.0)

    def _resolve_height_profile_ctrl(self) -> List[float]:
        scalar_profile = self._resolve_scalar_profile(
            [
                self._config.height_profile_ctrl_0,
                self._config.height_profile_ctrl_1,
                self._config.height_profile_ctrl_2,
                self._config.height_profile_ctrl_3,
            ]
        )
        return scalar_profile if scalar_profile else self._config.height_profile_ctrl

    def _resolve_height_profile_poly(self) -> List[float]:
        scalar_profile = self._resolve_scalar_profile(
            [
                self._config.height_profile_poly_c2,
                self._config.height_profile_poly_c4,
                self._config.height_profile_poly_c6,
            ]
        )
        return scalar_profile if scalar_profile else self._config.height_profile_poly

    def _resolve_grin_profile_ctrl(self) -> List[float]:
        scalar_profile = self._resolve_scalar_profile(
            [
                self._config.grin_profile_ctrl_0,
                self._config.grin_profile_ctrl_1,
                self._config.grin_profile_ctrl_2,
                self._config.grin_profile_ctrl_3,
            ]
        )
        if scalar_profile:
            return scalar_profile
        return self._config.grin_profile_ctrl or []

    def _resolve_scalar_profile(
        self, values: List[Optional[float]]
    ) -> List[float]:
        filtered = [value for value in values if value is not None]
        return filtered

    def _interpolate_profile(
        self,
        values: Optional[List[float]],
        radial_bins: int,
        *,
        default_value: float,
        clamp_min: float = 0.0,
        clamp_max: Optional[float] = 1.0,
    ) -> torch.Tensor:
        if values:
            profile = torch.tensor(values, dtype=torch.float64)
        else:
            profile = torch.full((radial_bins,), default_value, dtype=torch.float64)

        if profile.numel() == 1:
            profile = profile.repeat(radial_bins)
        elif profile.numel() != radial_bins:
            profile = torch.nn.functional.interpolate(
                profile.view(1, 1, -1),
                size=radial_bins,
                mode="linear",
                align_corners=True,
            ).view(-1)

        if clamp_max is None:
            return torch.clamp(profile, min=clamp_min)
        return torch.clamp(profile, min=clamp_min, max=clamp_max)

    def _precompute_psfs(self):
        """This will precompute the PSFs for all depths. This is done to avoid
        recomputing the PSF for each render call."""
        for depth in self._depths:
            self._psfs[depth.item()] = self._calculate_psf(depth).to(device)

    def _calculate_psf(self, depth: torch.Tensor):
        # electric field originating from point source
        u1 = torch.exp(self._k * torch.sqrt(self._X1_Y1 + depth.square()))

        # electric field at the aperture
        u2 = torch.mul(u1, self._pupil)

        # electric field at the sensor plane
        # Calculate the sqrt of the PSF
        u2_fft = torch.fft.fft2(torch.fft.fftshift(u2))
        H_u2_fft = torch.mul(torch.fft.fftshift(self._H), u2_fft)
        u3: torch.Tensor = torch.fft.ifftshift(torch.fft.ifft2(H_u2_fft))

        # Normalize the PSF by channel
        psf: torch.Tensor = u3.abs().square()
        psf = self._resize(psf)
        psf /= psf.sum(axis=(1, 2)).reshape(-1, 1, 1)

        # TODO: we have to do this post-calculations otherwise there are differences
        # between previous algo
        psf = psf.float()

        return psf

    def step(
        self, obs: Tuple[torch.Tensor, torch.Tensor] | None = None
    ) -> torch.Tensor:
        """Overwrites the default render method to apply the depth invariant PSF to the
        image."""
        if obs is not None:
            image, depth = obs
        else:
            image, depth = self._renderer.render()

        # Calculate the depth. Remove the sky depth, which is capped at the extent
        # of the configured environment and apply a far field approximation assumption.
        depth = depth[depth < torch.max(depth)]
        depth = torch.clamp(depth, 5 * max(self.config.focal), None)
        mean_depth = torch.mean(depth)

        # Add noise to the image
        image = self._apply_noise(image, self._config.noise_std)

        # Apply the depth invariant PSF
        psf = self._get_psf(mean_depth)

        # Image may be batched in the form
        image = image.permute(2, 0, 1).unsqueeze(0)
        psf = psf.unsqueeze(1)
        image = torch.nn.functional.conv2d(image, psf, padding="same", groups=3)

        # Apply the scaling intensity ratio
        if self._config.scale_intensity:
            image *= self._scaling_intensity

        # Post-process the image
        image = image.squeeze(0).permute(1, 2, 0)
        image = self._crop(image)
        image = torch.clip(image, 0, 1)

        return super().step(obs=image)

    def _apply_noise(self, image: torch.Tensor, std: float) -> torch.Tensor:
        """Add Gaussian noise to the image."""
        if std == 0.0:
            return image

        noise = torch.normal(mean=0.0, std=std, size=image.shape, device=device)
        return torch.clamp(image + noise, 0, 1)

    def _get_psf(self, depth: torch.Tensor) -> torch.Tensor:
        """This will retrieve the psf with the closest depth to the specified depth.
        If the psfs are precomputed, this will be a simple lookup. Otherwise, the psf
        will be calculated on the fly."""
        if self._psfs:
            closest_depth = self._depths[torch.argmin(torch.abs(depth - self._depths))]
            return self._psfs[closest_depth.item()]
        else:
            return self._calculate_psf(depth)

    def _crop(self, image: torch.Tensor) -> torch.Tensor:
        """Crop the image to the resolution specified in the config. This method
        supports input shape [W, H, 3]. It crops the center part of the image.
        """
        width, height, _ = image.shape
        target_width, target_height = self._config.resolution
        top = (height - target_height) // 2
        left = (width - target_width) // 2
        return image[left : left + target_width, top : top + target_height, :]

    def _resize(self, psf: torch.Tensor) -> torch.Tensor:
        """Resize the PSF to the psf_resolution."""
        return torch.nn.functional.interpolate(
            psf.unsqueeze(0),
            size=self._psf_resolution,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
