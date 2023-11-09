import asyncio
import logging
import math
import numpy as np

from nanosam.utils.predictor import Predictor

from PIL import Image
from threading import Lock
from typing import cast, ClassVar, List, Mapping, Optional, Sequence, Tuple, Union
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
    underlying: Camera
    properties: Camera.Properties
    predictor: Predictor

    @classmethod
    def new(cls, config: ComponentConfig, dependencies: Mapping[ResourceName, ResourceBase]) -> Self:
        cam = cls(config.name)
        cam.logger = getLogger(f'{__name__}.{cam.__class__.__name__}')
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
        cam_name = struct_to_dict(config.attributes).get("source")
        actual_cam = dependencies[Camera.get_resource_name(cam_name)]
        self.underlying = cast(Camera, actual_cam)
        self.properties = self.underlying.get_properties()
        #TODO: move image_encoder and mask_decoder to be args
        self.predictor = Predictor(
            "data/resnet18_image_encoder.engine",
            "data/mobile_sam_mask_decoder.engine"
        )
        return

    async def get_image(self, mime_type: str='', *, timeout: Optional[float]=None, **kwargs) -> Union[Image.Image, RawImage]:
        img = await self.underlying.get_image()
        self.predictor.set_image(img)

        # Segment using bounding box
        bbox = [100, 100, 850, 759]  # x0, y0, x1, y1
        points = np.array([
            [bbox[0], bbox[1]],
            [bbox[2], bbox[3]]
        ])
        point_labels = np.array([2, 3])
        mask, _, _ = self.predictor.predict(points, point_labels)
        mask = (mask[0, 0] > 0).detach().cpu().numpy()
        yellow_image = Image.new('RGBA', img.size, (255, 255, 0, 128))
        img.paste(yellow_image, (0, 0), mask)

        return img

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
