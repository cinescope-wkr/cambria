"""Wrapper around the mujoco viewer for rendering scenes."""

import atexit
import ctypes
from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Flag, auto
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type

import glfw
import imageio
import mujoco as mj
import numpy as np
import OpenGL.GL as GL
import torch
from hydra_config import HydraContainerConfig, HydraFlagWrapperMeta, config_wrapper

import cambrian.utils
from cambrian.renderer.overlays import MjCambrianViewerOverlay
from cambrian.utils.logger import get_logger
from cambrian.utils.spec import MjCambrianSpec

if TYPE_CHECKING:
    try:
        import mujoco.usd.exporter
    except ImportError:
        pass

has_pycuda_gl = False  # disable pycuda for now
try:
    if has_pycuda_gl:
        import pycuda.autoinit  # noqa
        import pycuda.driver as cuda
        import pycuda.gl as cudagl

    has_pycuda_gl = has_pycuda_gl
except ImportError:
    has_pycuda_gl = False

device = cambrian.utils.device
if has_pycuda_gl and torch.device(device) != torch.device("cuda"):
    get_logger().warning(
        "Not using CUDA device. Disabling PyCUDA GL interop for rendering."
    )
    has_pycuda_gl = False


class MjCambrianRendererSaveMode(Flag, metaclass=HydraFlagWrapperMeta):
    """The save modes for saving rendered images."""

    NONE = auto()
    GIF = auto()
    MP4 = auto()
    PNG = auto()
    WEBP = auto()
    USD = auto()


@config_wrapper
class MjCambrianRendererConfig(HydraContainerConfig):
    """The config for the renderer. Used for type hinting.

    A renderer corresponds to a single camera. The renderer can then view the scene in
    different ways, like offscreen (rgb_array) or onscreen (human).

    Attributes:
        render_modes (List[str]): The render modes to use for the renderer. See
            `MjCambrianRenderer.metadata["render.modes"]` for options.

        width (Optional[int]): The width of the rendered image. For onscreen renderers,
            if this is set, the window cannot be resized. Must be set for offscreen
            renderers.
        height (Optional[int]): The height of the rendered image. For onscreen
            renderers, if this is set, the window cannot be resized. Must be set for
            offscreen renderers.

        fullscreen (Optional[bool]): Whether to render in fullscreen or not. If True,
            the width and height are ignored and the window is rendered in fullscreen.
            This is only valid for onscreen renderers.

        scene (Type[mj.MjvScene]): The scene to render.
        scene_options (mj.MjvOption): The options to use for rendering.
        camera (mj.MjvCamera): The camera to use for rendering.

        use_shared_context (bool): Whether to use a shared context or not.
            If True, the renderer will share a context with other renderers. This is
            useful for rendering multiple renderers at the same time. If False, the
            renderer will create its own context. This is computationally expensive if
            there are many renderers.

        save_mode (Optional[MjCambrianRendererSaveMode]): The save modes to use for
            saving the rendered images. See `MjCambrianRenderer.SaveMode` for options.
            Must be set if `save` is called without save modes passed directly.
    """

    render_modes: List[str]

    width: Optional[int] = None
    height: Optional[int] = None

    fullscreen: Optional[bool] = None

    scene: Type[mj.MjvScene]
    scene_options: mj.MjvOption
    camera: mj.MjvCamera

    use_shared_context: bool

    save_mode: Optional[MjCambrianRendererSaveMode] = None


# ===================

GL_CONTEXT: mj.gl_context.GLContext = None
MJR_CONTEXT: mj.MjrContext = None
CUDA_CONTEXT: "cuda.Context" = None
if has_pycuda_gl:
    CUDA_CONTEXT = cuda.Device(0).make_context()


def free_contexts():
    global GL_CONTEXT, MJR_CONTEXT, CUDA_CONTEXT
    if GL_CONTEXT is not None:
        try:
            GL_CONTEXT.free()
        except Exception:
            pass
        finally:
            GL_CONTEXT = None
    if MJR_CONTEXT is not None:
        try:
            MJR_CONTEXT.free()
        except Exception:
            pass
        finally:
            MJR_CONTEXT = None
    if CUDA_CONTEXT is not None:
        try:
            CUDA_CONTEXT.detach()
        except Exception:
            pass
        finally:
            CUDA_CONTEXT = None


# Remove the automatic freeing. Will error out when calling free
# since we explicitly free the contexts in the atexit function.
mj.gl_context.GLContext.__del__ = lambda _: None


atexit.register(free_contexts)

# ===================


class MjCambrianViewer(ABC):
    """The base class for the viewer. This class should not be instantiated directly.

    Args:
        config (MjCambrianRendererConfig): The config to use for the viewer.
    """

    def __init__(self, config: MjCambrianRendererConfig):
        self._config = config

        self._spec: MjCambrianSpec = None
        self._viewport: mj.MjrRect = None
        self._scene: mj.MjvScene = None
        self._scene_options: mj.MjvOption = None
        self._camera: mj.MjvCamera = None

        self._gl_context: mj.gl_context.GLContext = None
        self._mjr_context: mj.MjrContext = None
        self._font = mj.mjtFontScale.mjFONTSCALE_50
        self._pixel_bytes = 3

        self._rgb_float32: torch.Tensor = None
        self._depth: torch.Tensor = None

        if has_pycuda_gl:
            self._rgb_pbo: int = None
            self._rgb_res: cudagl.RegisteredBuffer = None
            self._rgb_mapped_res: cuda.DeviceAllocation = None
            self._rgb_ptr: int = None

            self._depth_pbo: int = None
            self._depth_res: cudagl.RegisteredBuffer = None
            self._depth_mapped_res: cuda.DeviceAllocation = None
            self._depth_ptr: int = None

    def reset(self, spec: MjCambrianSpec, width: int, height: int):
        self._spec = spec

        # Only create the scene once
        if self._scene is None:
            self._scene = self._config.scene(self._spec.model)
        self._scene_options = deepcopy(self._config.scene_options)
        self._camera = deepcopy(self._config.camera)

        self._initialize_contexts(width, height)

        self._viewport = mj.MjrRect(0, 0, width, height)

        # Initialize the buffers
        if self._rgb_float32 is None or self._rgb_float32.shape != (
            height,
            width,
            self._pixel_bytes,
        ):
            self._rgb_uint8 = torch.zeros(
                (height, width, self._pixel_bytes),
                dtype=torch.uint8,
                device=device,
            )
            self._rgb_uint8_cpu = self._rgb_uint8.cpu()
            self._rgb_float32 = torch.zeros(
                (height, width, self._pixel_bytes),
                dtype=torch.float32,
                device=device,
            )
            self._depth = torch.zeros(
                (height, width),
                dtype=torch.float32,
                device=device,
            )
            self._depth_cpu = self._depth.cpu()

        if has_pycuda_gl:
            self._initialize_pbo()

    def _initialize_contexts(self, width: int, height: int):
        global GL_CONTEXT, MJR_CONTEXT

        # NOTE: All shared contexts must match either onscreen or offscreen. And their
        # height and width most likely must match as well. If the existing context
        # is onscreen and we're requesting offscreen, override use_shared_context (and
        # vice versa).
        use_shared_context = self._config.use_shared_context
        if use_shared_context and MJR_CONTEXT:
            if MJR_CONTEXT.currentBuffer != self.get_framebuffer_option():
                get_logger().warning(
                    "Overriding use_shared_context. "
                    "First buffer and current buffer don't match."
                )
                use_shared_context = False

        if use_shared_context:
            # Initialize or reuse the GL context
            GL_CONTEXT = GL_CONTEXT or mj.gl_context.GLContext(width, height)
            self._gl_context = GL_CONTEXT
            self.make_context_current()

            MJR_CONTEXT = MJR_CONTEXT or mj.MjrContext(self._spec.model, self._font)
            self._mjr_context = MJR_CONTEXT
        elif self._viewport is None or width != self.width or height != self.height:
            # If the viewport is None (i.e. this is the first reset), or the window
            # has been resized, create a new context. We'll need to clean up the old
            # context if it exists.
            if self._gl_context is not None:
                del self._gl_context
            if self._mjr_context is not None:
                del self._mjr_context

            # Initialize the new contexts
            self._gl_context = mj.gl_context.GLContext(width, height)
            self.make_context_current()
            self._mjr_context = mj.MjrContext(self._spec.model, self._font)
        self._mjr_context.readDepthMap = mj.mjtDepthMap.mjDEPTH_ZEROFAR
        mj.mjr_setBuffer(self.get_framebuffer_option(), self._mjr_context)

    def _initialize_pbo(self):
        assert has_pycuda_gl and torch.device(device) == torch.device("cuda")

        rgb_buffer_size = self.width * self.height * self._pixel_bytes
        depth_buffer_size = self.width * self.height

        self._rgb_pbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, int(self._rgb_pbo))
        GL.glBufferData(
            GL.GL_PIXEL_PACK_BUFFER, rgb_buffer_size, None, GL.GL_STREAM_READ
        )
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        self._rgb_res = cudagl.RegisteredBuffer(
            int(self._rgb_pbo), cuda.graphics_map_flags.READ_ONLY
        )
        self._rgb_mapped_res = self._rgb_res.map()
        self._rgb_ptr = self._rgb_mapped_res.device_ptr_and_size()[0]

        self._depth_pbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, int(self._depth_pbo))
        GL.glBufferData(
            GL.GL_PIXEL_PACK_BUFFER, depth_buffer_size, None, GL.GL_STREAM_READ
        )
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)
        self._depth_res = cudagl.RegisteredBuffer(
            int(self._depth_pbo), cuda.graphics_map_flags.READ_ONLY
        )
        self._depth_mapped_res = self._depth_res.map()
        self._depth_ptr = self._depth_mapped_res.device_ptr_and_size()[0]

    @abstractmethod
    def update(self, width: int, height: int):
        # Subclass should override this method such that this is not possible
        assert width == self._viewport.width and height == self._viewport.height

        mj.mjv_updateScene(
            self._spec.model,
            self._spec.data,
            self._scene_options,
            None,  # mjvPerturb
            self._camera,
            mj.mjtCatBit.mjCAT_ALL,
            self._scene,
        )

    def render(self, *, overlays: List[MjCambrianViewerOverlay] = []):
        self.make_context_current()
        self.update(self._viewport.width, self._viewport.height)

        # Reorder the overlays based on their layer type
        overlays = sorted(overlays, key=lambda overlay: overlay.layer.value)

        self._draw_before_render(overlays)

        mj.mjr_render(self._viewport, self._scene, self._mjr_context)

        self._draw_after_render(overlays)

    def _draw_before_render(self, overlays: List[MjCambrianViewerOverlay]):
        for overlay in overlays:
            overlay.draw_before_render(self._scene)

    def _draw_after_render(self, overlays: List[MjCambrianViewerOverlay]):
        if len(overlays) > 0:
            # Do a single mjr_overlay call to initialize underlying 2D rendering
            # If we don't do this, calling mjr_drawPixels without first calling
            # mjr_overlay will result in the pixels being drawn behind the 3D scene.
            mj.mjr_overlay(
                mj.mjtFont.mjFONT_NORMAL,
                mj.mjtGridPos.mjGRID_BOTTOMLEFT,
                self._viewport,
                "",
                "",
                self._mjr_context,
            )

        for overlay in overlays:
            overlay.draw_after_render(self._mjr_context, self._viewport)

    def read_pixels(
        self, read_rgb: bool, read_depth: bool
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        if has_pycuda_gl:
            out = self._read_pixels_cuda(read_rgb, read_depth)
        else:
            out = self._read_pixels(read_rgb, read_depth)
        return out

    def _read_pixels(
        self, read_rgb: bool, read_depth: bool
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        mj.mjr_readPixels(
            self._rgb_uint8_cpu if read_rgb else None,
            self._depth_cpu if read_depth else None,
            self._viewport,
            self._mjr_context,
        )

        if read_rgb:
            self._rgb_uint8.copy_(self._rgb_uint8_cpu, non_blocking=True)
            torch.divide(self._rgb_uint8, 255.0, out=self._rgb_float32)
        if read_depth:
            self._depth.copy_(self._depth_cpu, non_blocking=True)

        return self._rgb_float32 if read_rgb else None, (
            self._depth if read_depth else None
        )

    def _read_pixels_cuda(
        self, read_rgb: bool, read_depth: bool
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None]:
        mask = 0
        if read_depth:
            mask = GL.GL_COLOR_BUFFER_BIT
        if read_depth:
            mask |= GL.GL_DEPTH_BUFFER_BIT

        if self._spec.visual.quality.offsamples:
            # Multisampling
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._mjr_context.offFBO)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._mjr_context.offFBO_r)
            GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)
            GL.glBlitFramebuffer(
                self._viewport.left,
                self._viewport.bottom,
                self._viewport.left + self.width,
                self._viewport.bottom + self.height,
                self._viewport.left,
                self._viewport.bottom,
                self._viewport.left + self.width,
                self._viewport.bottom + self.height,
                mask,
                GL.GL_NEAREST,
            )
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._mjr_context.offFBO_r)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        else:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self._mjr_context.offFBO)
            GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)

        if read_rgb:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, int(self._rgb_pbo))
            GL.glReadPixels(
                self._viewport.left,
                self._viewport.bottom,
                self.width,
                self.height,
                self._mjr_context.readPixelFormat,
                GL.GL_UNSIGNED_BYTE,
                ctypes.c_void_p(0),
            )
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)

        if read_depth:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, int(self._depth_pbo))
            GL.glReadPixels(
                0,
                0,
                self.width,
                self.height,
                GL.GL_DEPTH_COMPONENT,
                GL.GL_FLOAT,
                ctypes.c_void_p(0),
            )
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, 0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._mjr_context.offFBO)
        GL.glReadBuffer(GL.GL_COLOR_ATTACHMENT0)
        GL.glDrawBuffer(GL.GL_COLOR_ATTACHMENT0)

        CUDA_CONTEXT.push()
        if read_rgb:
            cuda.memcpy_dtod(
                int(self._rgb_uint8.data_ptr()),
                self._rgb_ptr,
                self.width * self.height * self._pixel_bytes,
            )
            torch.divide(self._rgb_uint8, 255.0, out=self._rgb_float32)
        if read_depth:
            cuda.memcpy_dtod(
                int(self._depth.data_ptr()), self._depth_ptr, self.width * self.height
            )
        CUDA_CONTEXT.pop()

        return self._rgb_float32 if read_rgb else None, (
            self._depth if read_depth else None
        )

    @abstractmethod
    def make_context_current(self):
        pass

    @abstractmethod
    def get_framebuffer_option(self) -> int:
        pass

    @abstractmethod
    def is_running(self):
        pass

    # ===================

    @property
    def width(self) -> int:
        return self._viewport.width

    @width.setter
    def width(self, width: int):
        self._viewport.width = width

    @property
    def height(self) -> int:
        return self._viewport.height

    @height.setter
    def height(self, height: int):
        self._viewport.height = height

    @property
    def camera(self) -> mj.MjvCamera:
        return self._camera

    @property
    def scene_options(self) -> mj.MjvOption:
        return self._scene_options

    @property
    def config(self) -> MjCambrianRendererConfig:
        return self._config


class MjCambrianOffscreenViewer(MjCambrianViewer):
    """The offscreen viewer for rendering scenes."""

    def get_framebuffer_option(self) -> int:
        return mj.mjtFramebuffer.mjFB_OFFSCREEN.value

    def update(self, width: int, height: int):
        if self._viewport.width != width or self._viewport.height != height:
            self.make_context_current()
            self._viewport = mj.MjrRect(0, 0, width, height)
            mj.mjr_resizeOffscreen(width, height, self._mjr_context)

        super().update(width, height)

    def make_context_current(self):
        assert (
            self._gl_context is not None
        ), "GL context is not initialized, did you call reset?"
        self._gl_context.make_current()

    def is_running(self):
        return True


class MjCambrianOnscreenViewer(MjCambrianViewer):
    """The onscreen viewer for rendering scenes."""

    def __init__(self, config: MjCambrianRendererConfig):
        super().__init__(config)

        self._window = None
        self.default_window_pos: Tuple[int, int] = None
        self._scale: float = None

        self._last_mouse_x: int = None
        self._last_mouse_y: int = None
        self._is_paused: bool = None
        self.custom_key_callback: Callable = None

    def reset(self, spec: MjCambrianSpec, width: int, height: int):
        self._last_mouse_x: int = 0
        self._last_mouse_y: int = 0
        self._is_paused: bool = False

        if self._window is None:
            self._initialize_window(width, height)
        glfw.set_window_size(self._window, width, height)
        self.fullscreen(self._config.fullscreen if self._config.fullscreen else False)

        super().reset(spec, width, height)

        window_width, _ = glfw.get_window_size(self._window)
        self._scale = width / window_width

        glfw.set_cursor_pos_callback(self._window, self._cursor_pos_callback)
        glfw.set_mouse_button_callback(self._window, self._mouse_button_callback)
        glfw.set_scroll_callback(self._window, self._scroll_callback)
        glfw.set_key_callback(self._window, self._key_callback)

        glfw.swap_interval(0)

    def _initialize_window(self, width: int, height: int):
        global GL_CONTEXT, MJR_CONTEXT

        if not glfw.init():
            raise Exception("GLFW failed to initialize.")

        gl_context = None
        if self._config.use_shared_context:
            from mujoco.glfw import GLContext as GLFWGLContext

            GLFWGLContext.__del__ = lambda _: None

            GL_CONTEXT = GL_CONTEXT or GLFWGLContext(width, height)
            assert isinstance(GL_CONTEXT, GLFWGLContext), (
                f"The mujoco gl context must be of type {GLFWGLContext} to use "
                f"the OnscreenViewer, but got {type(GL_CONTEXT)} instead. "
                "Set the env variable `MUJOCO_GL` to `glfw` to use the correct context."
            )
            gl_context = GL_CONTEXT._context
        self._window = glfw.create_window(width, height, "MjCambrian", None, gl_context)
        if not self._window:
            glfw.terminate()
            raise Exception("GLFW failed to create window.")

        glfw.show_window(self._window)

        self.default_window_pos = glfw.get_window_pos(self._window)

    def make_context_current(self):
        glfw.make_context_current(self._window)
        super().make_context_current()

    def get_framebuffer_option(self) -> int:
        return mj.mjtFramebuffer.mjFB_WINDOW.value

    def update(self, width: int, height: int):
        if self._viewport.width != width or self._viewport.height != height:
            self.make_context_current()
            self._viewport = mj.MjrRect(0, 0, width, height)
            GL.glViewport(0, 0, width, height)

        super().update(width, height)

    def render(self, *, overlays: List[MjCambrianViewerOverlay] = []):
        if self._window is None:
            get_logger().warning("Tried to render destroyed window.")
            return
        elif glfw.window_should_close(self._window):
            get_logger().warning("Tried to render closed or closing window.")
            return

        while True:
            self.make_context_current()
            width, height = glfw.get_framebuffer_size(self._window)
            self._viewport = mj.MjrRect(0, 0, width, height)

            super().render(overlays=overlays)

            glfw.swap_buffers(self._window)
            glfw.poll_events()

            if not self._is_paused:
                break

    def is_running(self):
        return not (self._window is None or glfw.window_should_close(self._window))

    # ===================

    def fullscreen(self, fullscreen: bool):
        if self._window is None:
            get_logger().warning("Tried to set fullscreen to destroyed window.")
            return

        if fullscreen:
            monitor = glfw.get_primary_monitor()
            video_mode = glfw.get_video_mode(monitor)
            glfw.set_window_monitor(
                self._window,
                monitor,
                0,
                0,
                video_mode.size.width,
                video_mode.size.height,
                video_mode.refresh_rate,
            )

    # ===================

    def _cursor_pos_callback(self, window, xpos, ypos):
        left_button_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT)
        right_button_pressed = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT)
        if not (left_button_pressed or right_button_pressed):
            return

        shift_pressed = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )
        if right_button_pressed:
            MOVE_H, MOVE_V = mj.mjtMouse.mjMOUSE_MOVE_H, mj.mjtMouse.mjMOUSE_MOVE_V
            action = MOVE_H if shift_pressed else MOVE_V
        elif left_button_pressed:
            ROT_H, ROT_V = mj.mjtMouse.mjMOUSE_ROTATE_H, mj.mjtMouse.mjMOUSE_ROTATE_V
            action = ROT_H if shift_pressed else ROT_V
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        dx = int(self._scale * xpos) - self._last_mouse_x
        dy = int(self._scale * ypos) - self._last_mouse_y
        width, height = glfw.get_framebuffer_size(window)
        reldx, reldy = dx / width, dy / height

        mj.mjv_moveCamera(
            self._spec.model, action, reldx, reldy, self._scene, self._camera
        )

        self._last_mouse_x = int(self._scale * xpos)
        self._last_mouse_y = int(self._scale * ypos)

    def _mouse_button_callback(self, window, button, action, mods):
        x, y = glfw.get_cursor_pos(window)
        self._last_mouse_x = int(self._scale * x)
        self._last_mouse_y = int(self._scale * y)

    def _scroll_callback(self, window, xoffset, yoffset):
        mj.mjv_moveCamera(
            self._spec.model,
            mj.mjtMouse.mjMOUSE_ZOOM,
            0,
            -0.05 * yoffset,
            self._scene,
            self._camera,
        )

    def _key_callback(self, window, key, scancode, action, mods):
        if action != glfw.RELEASE:
            return

        # Close window.
        if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
            glfw.set_window_should_close(window, True)
            self._is_paused = False  # unpause so the window can close

        # Switch cameras
        if key == glfw.KEY_TAB:
            self._camera.fixedcamid += 1
            self._camera.type = mj.mjtCamera.mjCAMERA_FIXED
            if self._camera.fixedcamid >= self._spec.model.ncam:
                self._camera.fixedcamid = -1
                self._camera.type = mj.mjtCamera.mjCAMERA_FREE

        # Pause simulation
        if key == glfw.KEY_SPACE:
            self._is_paused = not self._is_paused

        # Screenshot
        if key == glfw.KEY_S:
            rgb, _ = self._read_pixels(read_rgb=True, read_depth=False)
            rgb = (rgb * 255.0).to(torch.uint8).cpu().numpy()
            rgb = np.flipud(rgb)
            imageio.imwrite("screenshot.png", rgb)
            get_logger().info("Saved screenshot at screenshot.png.")

        # Custom key callback
        if self.custom_key_callback is not None:
            self.custom_key_callback(window, key, scancode, action, mods)


class MjCambrianRenderer:
    """The renderer for rendering scenes. This is essentially a wrapper around the
    mujoco viewer/renderer.

    Args:
        config (MjCambrianRendererConfig): The config to use for the renderer.

    Attributes:
        metadata (Dict[str, List[str]]): The metadata for the renderer. The render modes
            are stored here.
    """

    metadata: Dict[str, List[str]] = {
        "render.modes": ["human", "rgb_array", "depth_array"]
    }

    def __init__(self, config: MjCambrianRendererConfig):
        self._config = config
        self._spec: MjCambrianSpec = None

        assert all(
            mode in self.metadata["render.modes"] for mode in self._config.render_modes
        ), f"Invalid render mode found. Valid modes are {self.metadata['render.modes']}"

        self._viewer: MjCambrianViewer = None
        if "human" in self._config.render_modes:
            self._viewer = MjCambrianOnscreenViewer(self._config)
        else:
            self._viewer = MjCambrianOffscreenViewer(self._config)

        self._record: bool = False
        self._rgb_buffer: List[torch.Tensor] = []
        self._usd_exporter: Optional["mujoco.usd.exporter.USDExporter"] = None
        self._should_render: bool = any(
            m in self._config.render_modes for m in ["rgb_array", "depth_array"]
        )
        self._return_rgb: bool = "rgb_array" in self._config.render_modes
        self._return_depth: bool = "depth_array" in self._config.render_modes

    def reset(
        self,
        spec: MjCambrianSpec,
        width: Optional[int] = None,
        height: Optional[int] = None,
    ) -> torch.Tensor | None:
        self._spec = spec

        width = width or self._config.width or spec.model.vis.global_.offwidth
        height = height or self._config.height or spec.model.vis.global_.offheight

        if width > spec.model.vis.global_.offwidth:
            spec.model.vis.global_.offwidth = width
        if height > spec.model.vis.global_.offheight:
            spec.model.vis.global_.offheight = height

        self._viewer.reset(spec, width, height)

        return self.render(resetting=True)

    def render(
        self, *, overlays: List[MjCambrianViewerOverlay] = [], resetting: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor] | None:
        self._viewer.render(overlays=overlays)

        if not self._should_render:
            return

        rgb, depth = self._viewer.read_pixels(
            read_rgb=self._return_rgb, read_depth=self._return_depth
        )

        if self._record and not resetting:
            rgb_to_record = rgb.clone() if rgb.device == "cpu" else rgb.cpu().clone()
            self._rgb_buffer.append(rgb_to_record)

        returns = []
        if self._return_rgb:
            returns.append(rgb)

        if self._return_depth:
            returns.append(depth)

        if self._usd_exporter:
            self._usd_exporter.update_scene(self._spec.data, self._viewer.scene_options)

        return returns if len(returns) > 1 else returns[0]

    def is_running(self):
        return self._viewer.is_running()

    # ===================

    def save(
        self,
        path: Path | str,
        *,
        save_mode: Optional[MjCambrianRendererSaveMode] = None,
        fps: int = 50,
    ):
        save_mode = save_mode or self._config.save_mode
        duration = 1000 / fps

        if not self._record:
            get_logger().warning(
                "Not recording. Check if you called record() "
                "or `rgb_array` is in render_modes. Ignoring save..."
            )
            return
        assert len(self._rgb_buffer) > 0, "Cannot save empty buffer."

        get_logger().info(f"Saving visualizations at {path}...")

        path = Path(path)
        rgb_buffer = (
            (torch.stack(self._rgb_buffer) * 255.0).to(torch.uint8).cpu().numpy()
        )
        # Mujoco uses OpenGL, which uses bottom-left origin, so flip the buffer
        # since most of python uses top-left origin.
        rgb_buffer = np.flip(rgb_buffer, axis=1)

        if save_mode & MjCambrianRendererSaveMode.MP4:
            try:
                mp4 = path.with_suffix(".mp4")
                imageio.mimwrite(mp4, rgb_buffer, fps=fps)
            except TypeError:
                get_logger().error(
                    "imageio is not compiled with ffmpeg. "
                    "You may need to install it with `pip install imageio[ffmpeg]`."
                )
            get_logger().debug(f"Saved visualization at {mp4}")
        if save_mode & MjCambrianRendererSaveMode.PNG:
            png = path.with_suffix(".png")
            idx = -2 if len(rgb_buffer) > 1 else -1
            imageio.imwrite(png, rgb_buffer[idx])
            get_logger().debug(f"Saved visualization at {png}")
        if save_mode & MjCambrianRendererSaveMode.GIF:
            gif = path.with_suffix(".gif")
            imageio.mimwrite(gif, rgb_buffer, loop=0, duration=duration)
            get_logger().debug(f"Saved visualization at {gif}")
        if save_mode & MjCambrianRendererSaveMode.WEBP:
            webp = path.with_suffix(".webp")
            imageio.mimwrite(webp, rgb_buffer, fps=fps, lossless=True)
            get_logger().debug(f"Saved visualization at {webp}")
        if save_mode & MjCambrianRendererSaveMode.USD or self._usd_exporter:
            assert self._usd_exporter, "USD exporter not initialized."
            self._usd_exporter.save_scene("usd")

        get_logger().debug(f"Saved visualization at {path}")

    def record(
        self,
        record: bool = True,
        *,
        path: Optional[Path] = None,
        save_mode: Optional[MjCambrianRendererSaveMode] = None,
    ):
        get_logger().info(f"{'Starting' if record else 'Stopping'} recording...")
        if record and self._record:
            get_logger().warning("Already recording. Ignoring...")
            return
        elif not record and not self._record:
            get_logger().warning("Not recording. Ignoring...")
            return
        elif record and "rgb_array" not in self._config.render_modes:
            render_modes = list(self._config.render_modes)
            get_logger().warning(
                f"Cannot record without rgb_array mode: {render_modes}. Ignoring..."
            )
            return

        save_mode = save_mode or self._config.save_mode
        if record and MjCambrianRendererSaveMode.USD & save_mode:
            try:
                import mujoco.usd.exporter
            except ImportError:
                get_logger().error(
                    "MuJoCo wasn't installed with USD support. You may need to "
                    "run `pip install mujoco[usd]`."
                )
                return

            camera_names = [
                self._spec.get_camera_name(i) for i in range(self._spec.model.ncam)
            ]
            self._usd_exporter = mujoco.usd.exporter.USDExporter(
                self._spec.model,
                self.height,
                self.width,
                self._config.scene.maxgeom,
                output_directory_root=path,
                output_directory="usd",
                camera_names=camera_names,  # save all cameras
                verbose=False,
            )
        elif not record:
            self._rgb_buffer.clear()
            self._usd_exporter = None
        self._record = record

    # ===================

    @property
    def config(self) -> MjCambrianRendererConfig:
        return self._config

    @property
    def viewer(self) -> MjCambrianViewer:
        return self._viewer

    @property
    def width(self) -> int:
        return self._viewer.width

    @property
    def height(self) -> int:
        return self._viewer.height

    @property
    def ratio(self) -> float:
        return self.width / self.height
