import asyncio
import logging
import math
import numpy as np

from PIL.Image import Image
from threading import Lock
from typing import ClassVar, List, Mapping, Optional, Sequence, Tuple, Union
from typing_extensions import Self
from viam.components.camera import Camera, DistortionParameters, IntrinsicParameters, RawImage
from viam.logging import getLogger
from viam.media.video import NamedImage 
from viam.module.types import Reconfigurable
from viam.proto.app.robot import ComponentConfig
from viam.proto.common import ResourceName, ResponseMetadata
from viam.resource.base import ResourceBase
from viam.resource.registry import Registry, ResourceCreatorRegistration
from viam.resource.types import Model, ModelFamily
from viam.utils import struct_to_dict
from viam.media.video import CameraMimeType


class NanoSAM(Camera, Reconfigurable):
    MODEL: ClassVar[Model] = Model(ModelFamily('viam-soleng', 'nanoSAM'), 'segmenter-cam')
    logger: logging.Logger
    properties: Camera.Properties

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        cam = cls(config.name)
        cam.logger = getLogger(f'{__name__}.{cam.__class__.__name__}')
        #TODO: confirm intrinsics and distortion.
        lidar.properties = Camera.Properties(
            supports_pcd=False,
            intrinsic_parameters=IntrinsicParameters(width_px=0, height_px=0, focal_x_px=0.0, focal_y_px=0.0, center_x_px=0.0),
            distortion_parameters=DistortionParameters(model='')
        )
        cam.reconfigure(config, dependencies)

        return cam

    def __del__(self):
        pass

    @classmethod
    def  validate_config(cls, config: ComponentConfig) -> Sequence[str]:
        attributes_dict = struct_to_dict(config.attributes)
        source = attributes_dict.get("source", "")
        assert isinstance(source, str)
        if source == "":
            raise Exception("the source argument is required and should contain the name of the camera component the segmenter-cam should segment the images of.")
        return []

    def reconfigure(self, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]):
        pass

    async def get_image(self, mime_type: str='', *, timeout: Optional[float]=None, **kwargs) -> Union[Image, RawImage]:
        #TODO: do the work here
        raise NotImplementedError()

    async def get_images(self, *, timeout: Optional[float]=None, **kwargs) -> Tuple[List[NamedImage], ResponseMetadata]:
        raise NotImplementedError()

    async def get_point_cloud(self, *, timeout: Optional[float]=None, **kwargs) -> Tuple[bytes, str]:
        raise NotImplementedError()

    async def get_properties(self, *, timeout: Optional[float] = None, **kwargs) -> Camera.Properties:
        return self.properties

Registry.register_resource_creator(
    Camera.SUBTYPE,
    NanoSAM.MODEL,
    ResourceCreatorRegistration(NanoSAM.new, NanoSAM.validate_config)
)
