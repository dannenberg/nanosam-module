import asyncio
from nano_sam import NanoSAM

from viam.components.camera import Camera
from viam.module.module import Module
from viam.resource.registry import Registry, ResourceCreatorRegistration

async def main():
    module = Module.from_args()
    module.add_model_from_registry(Camera.SUBTYPE, NanoSAM.MODEL)
    await module.start()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(e)
