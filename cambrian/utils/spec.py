"""Wrapper for the Mujoco MjModel and MjData classes."""

from typing import Any, Self

import mujoco as mj

from cambrian.utils.cambrian_xml import MjCambrianXML


class _MjCambrianSpec:
    def __init__(self, spec: mj.MjSpec):
        self._spec: mj.MjSpec = spec

        self._model: mj.MjModel = None
        self._data: mj.MjData = None

        self._model = self._spec.compile()
        self._data = mj.MjData(self._model)
        self.compile()

    def compile(self) -> Self:
        return self.recompile()

    def recompile(self) -> Self:
        self._model, self._data = self._spec.recompile(self._model, self._data)
        return self

    # ======================

    def _get_id(self, obj_type: int, obj_name: str) -> int:
        return mj.mj_name2id(self.model, obj_type, obj_name)

    def _get_name(self, obj_type: int, obj_adr: int) -> str:
        return mj.mj_id2name(self.model, obj_type, obj_adr)

    def get_body_id(self, body_name: str) -> int:
        """Get the ID of a Mujoco body."""
        return self._get_id(mj.mjtObj.mjOBJ_BODY, body_name)

    def get_body_name(self, bodyadr: int) -> str:
        """Get the name of a Mujoco body."""
        return self._get_name(mj.mjtObj.mjOBJ_BODY, bodyadr)

    def get_geom_id(self, geom_name: str) -> int:
        """Get the ID of a Mujoco geometry."""
        return self._get_id(mj.mjtObj.mjOBJ_GEOM, geom_name)

    def get_geom_name(self, geomadr: int) -> str:
        """Get the name of a Mujoco geometry."""
        return self._get_name(mj.mjtObj.mjOBJ_GEOM, geomadr)

    def get_site_id(self, site_name: str) -> int:
        """Get the ID of a Mujoco geometry."""
        return self._get_id(mj.mjtObj.mjOBJ_SITE, site_name)

    def get_site_name(self, siteadr: int) -> str:
        """Get the name of a Mujoco geometry."""
        return self._get_name(mj.mjtObj.mjOBJ_SITE, siteadr)

    def get_joint_id(self, joint_name: str) -> int:
        """Get the ID of a Mujoco geometry."""
        return self._get_id(mj.mjtObj.mjOBJ_JOINT, joint_name)

    def get_joint_name(self, jointadr: int) -> str:
        """Get the name of a Mujoco geometry."""
        return self._get_name(mj.mjtObj.mjOBJ_JOINT, jointadr)

    def get_camera_id(self, camera_name: str) -> int:
        """Get the ID of a Mujoco camera."""
        return self._get_id(mj.mjtObj.mjOBJ_CAMERA, camera_name)

    def get_camera_name(self, cameraadr: int) -> str:
        """Get the name of a Mujoco camera."""
        return self._get_name(mj.mjtObj.mjOBJ_CAMERA, cameraadr)

    def get_light_id(self, light_name: str) -> int:
        """Get the ID of a Mujoco light."""
        return self._get_id(mj.mjtObj.mjOBJ_LIGHT, light_name)

    def get_light_name(self, lightadr: int) -> str:
        """Get the name of a Mujoco light."""
        return self._get_name(mj.mjtObj.mjOBJ_LIGHT, lightadr)

    def get_sensor_id(self, sensor_name: str) -> int:
        """Get the ID of a Mujoco sensor."""
        return self._get_id(mj.mjtObj.mjOBJ_SENSOR, sensor_name)

    def get_sensor_name(self, sensoradr: int) -> str:
        """Get the name of a Mujoco sensor."""
        return self._get_name(mj.mjtObj.mjOBJ_SENSOR, sensoradr)

    def get_material_id(self, material_name: str) -> int:
        """Get the ID of a Mujoco material."""
        return self._get_id(mj.mjtObj.mjOBJ_MATERIAL, material_name)

    def get_material_name(self, materialadr: int) -> str:
        """Get the name of a Mujoco material."""
        return self._get_name(mj.mjtObj.mjOBJ_MATERIAL, materialadr)

    # ======================

    def save(self, filename: str):
        MjCambrianXML.from_string(self._spec.to_xml()).write(filename)

    # ======================

    def __getattr__(self, name: str) -> Any:
        return getattr(self._spec, name)

    @property
    def model(self) -> mj.MjModel:
        return self._model

    @property
    def data(self) -> mj.MjData:
        return self._data


MjCambrianSpec = _MjCambrianSpec | mj.MjSpec


def spec_from_xml(xml: MjCambrianXML) -> MjCambrianSpec:
    return _MjCambrianSpec(xml.to_spec())


def spec_from_xml_string(xml: str) -> MjCambrianSpec:
    return spec_from_xml(MjCambrianXML.from_string(xml))
