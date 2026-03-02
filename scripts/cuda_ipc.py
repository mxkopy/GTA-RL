import cupy
from structs import CUDAPitchedArrayObject
from ipc import StructuredMemory

class CUDAArray:

    @staticmethod
    def get_dtype(metadata: CUDAPitchedArrayObject):
        print(metadata)
        if metadata.format.f == 1:
            return cupy.uint8
        if metadata.format.f == 2:
            return cupy.float32

    def init_cuda_array(self, metadata: CUDAPitchedArrayObject):
        memory_handle = cupy.cuda.runtime.ipcOpenMemHandle(metadata.handle)
        mem_buffer = cupy.cuda.UnownedMemory(memory_handle, metadata.pitch * metadata.extent.height, owner=self, device_id=0)
        memory_pointer = cupy.cuda.MemoryPointer(mem_buffer, 0)
        channel_format = (metadata.format.x, metadata.format.y, metadata.format.z, metadata.format.w)
        components = sum(x > 0 for x in channel_format)
        bpp = sum(x for x in channel_format) // 8
        return cupy.ndarray(shape=(metadata.extent.height, metadata.pitch // bpp, components), dtype=CUDAArray.get_dtype(metadata), memptr=memory_pointer)

    def __init__(self, tagname, *args, **kwargs):
        self.metadata_memory = StructuredMemory(tagname)
        self.cuda_array = self.init_cuda_array(self.metadata_memory.data)
    
    @property
    def data(self):
        return self.cuda_array
    
