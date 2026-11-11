"""Offline native loader; CUDA requires explicit real Root ownership and held FDs.

No admission/resource observer is invented by this product. The executor supplies
truthful ownership/lifetime validation; the historical dummy-UUID guard is intact
in its original namespace. CPU remains explicit; full native execution never uses automatic placement.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
import os,stat,uuid as uuid_module

_OWNER=ContextVar('ripple_native95_cuda_owner',default=None)

@dataclass(frozen=True)
class HeldLease:
    path:str
    fd:int

@dataclass(frozen=True)
class OwnedCUDAAdmission:
    device:str
    physical_uuid:str
    owner_receipt:dict
    physical_shared:tuple
    role_exclusive:HeldLease
    check_live:object

    def check(self):
        if (not isinstance(self.device,str)or not self.device.startswith('cuda:')or not self.device[5:].isdigit()
            or not isinstance(self.physical_uuid,str)or not self.physical_uuid.startswith('GPU-')
            or not callable(self.check_live)):raise ValueError('explicit owned CUDA device/UUID/lifetime checker required')
        r=self.owner_receipt
        if (not isinstance(r,dict)or set(r)!={'path','bytes','sha256'}or not Path(r['path']).is_absolute()
            or type(r['bytes'])is not int or r['bytes']<1 or not isinstance(r['sha256'],str)or len(r['sha256'])!=64):raise ValueError('actual owner receipt FullRecord required')
        if not isinstance(self.physical_shared,tuple)or len(self.physical_shared)!=2:raise ValueError('both physical shared leases required')
        leases=(*self.physical_shared,self.role_exclusive)
        if any(not isinstance(x,HeldLease)for x in leases):raise ValueError('original held lease descriptors required')
        if len({x.path for x in leases})!=3 or len({x.fd for x in leases})!=3:raise ValueError('distinct two physical SH and role EX leases required')
        for lease in leases:
            if not isinstance(lease,HeldLease)or type(lease.fd)is not int or lease.fd<0 or not Path(lease.path).is_absolute():raise ValueError('literal held lease descriptor required')
            actual=os.fstat(lease.fd);named=os.stat(lease.path)
            if not stat.S_ISREG(actual.st_mode)or(actual.st_dev,actual.st_ino)!=(named.st_dev,named.st_ino):raise ValueError('held original lease FD/path changed')
        # FD identity alone cannot prove external policy ownership or lock mode.
        # The executor's genuine check must cover SH/EX, caps/floors and lifetime.
        if self.check_live()is False:raise RuntimeError('actual owned lifetime refused')
        import torch
        actual=torch.cuda.get_device_properties(int(self.device[5:])).uuid
        def canonical(value):
            if isinstance(value,str):return 'GPU-'+str(uuid_module.UUID(value[4:]if value.startswith('GPU-')else value))
            return 'GPU-'+str(uuid_module.UUID(bytes=bytes(value.bytes)))
        if canonical(actual)!=canonical(self.physical_uuid):raise RuntimeError('actual physical CUDA device differs from admitted owner')

def require_owner(device):
    device=str(device)
    if device=='cpu':return None
    owner=_OWNER.get()
    if not isinstance(owner,OwnedCUDAAdmission)or owner.device!=device:raise ValueError('CUDA requires explicit OwnedCUDAAdmission; no auto/idle-device fallback')
    owner.check();return owner

@contextmanager
def admitted(owner):
    if not isinstance(owner,OwnedCUDAAdmission):raise TypeError('OwnedCUDAAdmission required')
    owner.check();token=_OWNER.set(owner)
    try:yield owner
    finally:_OWNER.reset(token)
