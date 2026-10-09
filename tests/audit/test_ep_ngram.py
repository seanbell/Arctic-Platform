import sys
import tempfile
from datetime import timedelta
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
torch.set_num_threads(1)
def verdict(ok, details):
    print(details, flush=True)
    assert ok, details
import torch.distributed as dist
import torch.multiprocessing as mp
from arctic_platform.model.implementations.qwen38.modeling_qwen4_exp import EPShardedEmbedding

def worker(rank, rendezvous, output):
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=2, timeout=timedelta(seconds=15))
    table = torch.tensor([[10.], [20.], [30.], [40.]])
    layer = EPShardedEmbedding(4, 1)
    layer.weight = torch.nn.Parameter(table[rank*2:rank*2+2].clone(), requires_grad=False)
    layer._ep_rank, layer._ep_world_size, layer._ep_group = rank, 2, dist.group.WORLD
    same = layer(torch.tensor([0]))
    assert torch.equal(same, table[[0]])
    if rank == 0:
        print('positive control: identical rank IDs PASS', flush=True)
    ids = torch.tensor([rank*2])
    actual = layer(ids)
    Path(output, str(rank)).write_text(str(float(actual.item())))
    print(f'rank={rank} id={ids.item()} expected={table[ids].item()} actual={actual.item()}', flush=True)
    if '--unequal' in sys.argv:
        ids = torch.tensor([0] if rank == 0 else [2,3])
        actual = layer(ids)
        assert torch.equal(actual, table[ids]), (rank, actual, table[ids])
        print(f'unequal rank={rank} shape={list(ids.shape)} PASS', flush=True)
    dist.destroy_process_group()

if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(worker, args=(str(Path(tmp)/'gloo'), tmp), nprocs=2, join=True)
        values = [float(Path(tmp,str(rank)).read_text()) for rank in range(2)]
        verdict(values == [10.,30.], f'independent rank values={values}')
