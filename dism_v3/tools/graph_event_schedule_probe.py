"""Opt-in scheduling ablations; never imported by production dispatch.

All variants call the original arithmetic helpers. Direct pointers and task
grids are frozen to a completed plan and are diagnostic controls, not a dynamic
planner implementation. The original acquire gate remains before data access.
"""
import torch
import triton as tr
import triton.language as tl

from flash_dism.kernels.planner_graph import acquire, pointer, tau_stage
from flash_dism.kernels.hard_backward_fused import (
    _chunk_backward, _chunk_summaries, _summary_scan,
)


@tr.jit
def task_body(task, SQ, SK, V, DO, DQ, DK, DV_OUT,
              rows, prefix, weight, dlog, length, offsets, reset, ids, owners,
              h, g, a, tau, C: tl.constexpr, STAGE: tl.constexpr):
    if STAGE == 0:
        _chunk_backward(SQ, SK, V, DO, DQ, DK, DV_OUT, tau, None,
            None, rows, prefix, weight, dlog, length, 32, 64, C, 32, 64, True,
            BC=C, LOCAL=1, TASK=task)
    elif STAGE == 1:
        _chunk_summaries(SQ, SK, V, DO, h, g, a, ids, rows, prefix,
            weight, length, reset, 32, 64, C, 32, 64, True, TASK=task)
    elif STAGE == 2:
        _summary_scan(h, g, a, offsets, 2048, 2048, TASK=task)
    else:
        _chunk_backward(SQ, SK, V, DO, DQ, DK, DV_OUT, tau, owners,
            offsets, rows, prefix, weight, dlog, length, 32, 64, C, 32, 64, True,
            H=h, G=g, CHUNKS=ids, RESET=reset, TASK=task)


@tr.jit
def probe_stage(D, Choice, SQ, SK, V, DO, DQ, DK, DV_OUT,
                rows, prefix, weight, dlog, length, offsets, reset, ids, owners,
                h, g, a, tau, C: tl.constexpr, STAGE: tl.constexpr,
                DIRECT: tl.constexpr, LOOP: tl.constexpr):
    if acquire(Choice) == 1:
        count = tl.load(D + (1 if STAGE == 2 else 0)).to(tl.int32)
        if not DIRECT:
            rows = pointer(D, 2, tl.int32)
            prefix, weight = pointer(D, 3, tl.float32), pointer(D, 4, tl.float32)
            dlog, length = pointer(D, 5, tl.float32), pointer(D, 6, tl.int32)
            offsets, reset = pointer(D, 8, tl.int64), pointer(D, 9, tl.int32)
            ids, owners = pointer(D, 10, tl.int32), pointer(D, 11, tl.int32)
            h, g, a = pointer(D, 12, tl.float32), pointer(D, 13, tl.float32), pointer(D, 14, tl.float32)
            tau = pointer(D, 15, tl.float32)
        if LOOP:
            for task in range(tl.program_id(0), count, tl.num_programs(0)):
                task_body(task, SQ, SK, V, DO, DQ, DK, DV_OUT,
                    rows, prefix, weight, dlog, length, offsets, reset, ids, owners,
                    h, g, a, tau, C, STAGE)
        else:
            task = tl.program_id(0)
            if task < count:
                task_body(task, SQ, SK, V, DO, DQ, DK, DV_OUT,
                    rows, prefix, weight, dlog, length, offsets, reset, ids, owners,
                    h, g, a, tau, C, STAGE)


class ScheduleProbe:
    def __init__(self, payload, descriptor, choice, *, direct=False, loop=True,
                 ctas=512, task_grid=False, skip_empty=False):
        self.payload, self.descriptor, self.choice = payload, descriptor, choice
        self.direct, self.loop, self.ctas, self.task_grid = direct, loop, ctas, task_grid
        self.skip_empty=skip_empty
        if not loop and not task_grid:
            raise ValueError('one-task control requires a covering grid')
        tensors=[x for x in payload.owners if isinstance(x,torch.Tensor) and x.is_cuda]
        for group in (*payload.device.buckets,payload.device.multi):
            tensors.extend(x for x in group.values() if isinstance(x,torch.Tensor))
        pointers={x.data_ptr():x for x in tensors if x.numel()}
        self.records=payload.descriptor.reshape(4,16).tolist()
        self.dummies={dtype:torch.empty(1,dtype=dtype,device=descriptor.device)
                      for dtype in (torch.int32,torch.int64,torch.float32)}
        fields=(2,3,4,5,6,8,9,10,11,12,13,14,15)
        self.args=[]
        for record in self.records:
            args=[]
            for index in fields:
                dtype=torch.int64 if index==8 else torch.int32 if index in (2,6,9,10,11) else torch.float32
                args.append(pointers[record[index]] if record[index] else self.dummies[dtype])
            self.args.append(args)
        self.last_kernels=[]

    @torch.no_grad()
    def backward(self,sq,sk,value,dout,*,heads):
        dq,dk,dv=[torch.zeros_like(x,dtype=torch.float32) for x in (sq,sk,value)]
        dt=torch.zeros(heads,dtype=torch.float32,device=sq.device)
        self.last_kernels=[]
        for i,width in enumerate((16,32,64,64)):
            if self.skip_empty and not self.records[i][0]:
                continue
            desc=self.descriptor[i*16:]
            # Tau grid stays fixed across these ablations.
            self.last_kernels.append(tau_stage[(self.ctas,)](desc,self.choice,dt,True,256))
            for stage in ((0,) if i<3 else (1,2,3)):
                count=self.records[i][1 if stage==2 else 0]
                grid=max(1,count) if self.task_grid else self.ctas
                self.last_kernels.append(probe_stage[(grid,1)](
                    desc,self.choice,sq,sk,value,dout,dq,dk,dv,*self.args[i],
                    width,stage,self.direct,self.loop,
                    num_warps=2 if width==16 else 4,num_stages=1))
            self.last_kernels.append(tau_stage[(self.ctas,)](desc,self.choice,dt,False,256))
        return dict(sq_vec=dq,sk_vec=dk,v=dv,rtau=dt)
