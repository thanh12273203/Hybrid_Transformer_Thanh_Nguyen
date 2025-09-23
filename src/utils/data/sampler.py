from typing import List, NamedTuple, Iterator

import torch
from torch.utils.data import Sampler


class SampleKey(NamedTuple):
    file_idx: int  # index into your dataset's file list
    event_idx: int  # 0 .. events_per_file-1


def _coprime_step(n: int, gen: torch.Generator) -> int:
    # n = 100_000 = 2^5 * 5^5, so choose step that is odd and not divisible by 5
    # This guarantees gcd(step, n) == 1.
    while True:
        k = int(torch.randint(0, n // 2, (1,), generator=gen).item())
        step = 2 * k + 1
        if step % 5 != 0:
            return step
        

class JetClassDistributedSampler(Sampler[List[SampleKey]]):
    """
    Yields batches where each batch touches exactly one file per class (10 files total).
    Within the 10 files, it samples disjoint event indices across (world_size * replicas_per_rank).
    __len__ == number of file-groups per epoch (usually 100 when each class has 100 files).

    Requirements
    ------------
    - batch_size % (world_size * 10) == 0
    - If replicas_per_rank > 1: batch_size % (world_size * replicas_per_rank * 10) == 0
    - Dataset __getitem__ must accept a SampleKey(file_idx, event_idx) or a tuple with same fields.

    Parameters
    ----------
    files_by_class: List[List[int]]
        List of length 10. Each element is a list of file indices for that class.
        All classes should have the same length (e.g., 100).
    events_per_file: int
        Number of events in each file.
    batch_size: int
        Global batch size (sum over all ranks and replicas).
    num_replicas: int
        torch.distributed.get_world_size() or 1 if not using DDP.
    rank: int
        torch.distributed.get_rank() or 0.
    replicas_per_rank: int
        Number of independent dataloaders per rank (<=4 as per user).
    local_replica_id: int
        0..replicas_per_rank-1 to disjointly split work on the same rank.
    seed: int
        Base seed for deterministic shuffles per epoch.
    shuffle_files: bool
        Shuffle file order per class each epoch.
    """
    def __init__(
        self,
        files_by_class: List[List[int]],
        events_per_file: int,
        batch_size: int,
        num_replicas: int = 1,
        rank: int = 0,
        replicas_per_rank: int = 1,
        local_replica_id: int = 0,
        seed: int = 42,
        shuffle_files: bool = True,
    ):
        assert len(files_by_class) == 10, "Expect exactly 10 classes."
        n_per_class = [len(x) for x in files_by_class]
        assert len(set(n_per_class)) == 1, "All classes must have the same file count."

        self.num_groups = n_per_class[0]  # e.g., 100
        self.files_by_class = files_by_class
        self.events_per_file = int(events_per_file)
        self.shuffle_files = bool(shuffle_files)
        self.num_replicas = int(num_replicas)
        self.rank = int(rank)
        self.replicas_per_rank = int(replicas_per_rank)
        self.local_replica_id = int(local_replica_id)
        self.virtual_world = self.num_replicas * self.replicas_per_rank
        self.virtual_rank = self.rank * self.replicas_per_rank + self.local_replica_id

        self.batch_size_global = int(batch_size)
        assert self.batch_size_global % (self.num_replicas * 10) == 0, f"batch_size must be divisible by {self.num_replicas * 10}."
        
        # Stronger guarantee if using replicas_per_rank > 1
        if self.replicas_per_rank > 1:
            assert self.batch_size_global % (self.virtual_world * 10) == 0, f"with replicas_per_rank > 1, batch_size must be divisible by {self.virtual_world * 10}."

        # Per (virtual) rank batch size
        self.local_batch_size = self.batch_size_global // self.virtual_world
        assert self.local_batch_size % 10 == 0, "local_batch_size must be divisible by 10."
        self.per_file_local = self.local_batch_size // 10  # items drawn from each of the 10 files locally

        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_groups

    def _file_orders_for_epoch(self) -> List[List[int]]:
        # Optionally shuffle the files per class; deterministic per epoch.
        if not self.shuffle_files:
            return [list(range(len(cls_files))) for cls_files in self.files_by_class]
        
        orders = []
        for c in range(10):
            n = len(self.files_by_class[c])
            g = torch.Generator()
            g.manual_seed(self.seed * 1009 + self.epoch * 7919 + c * 271)
            orders.append(torch.randperm(n, generator=g).tolist())

        return orders

    def __iter__(self) -> Iterator[List[SampleKey]]:
        orders = self._file_orders_for_epoch()

        # For each group, pick one file per class
        for group_idx in range(self.num_groups):
            selected_files = [
                self.files_by_class[c][orders[c][group_idx]] for c in range(10)
            ]

            # Build local batch by taking per_file_local indices from each selected file
            batch = []

            for fid in selected_files:
                # Deterministic per (epoch, group, file) to keep all ranks in sync
                g = torch.Generator()

                # Mix seeds; different constants reduce correlation
                g.manual_seed(
                    self.seed
                    ^ (self.epoch * 0x9E3779B97F4A7C15 & 0xFFFFFFFFFFFF)
                    ^ (group_idx * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFF)
                    ^ (fid * 0x94D049BB133111EB & 0xFFFFFFFFFFFF)
                )
                N = self.events_per_file
                start = int(torch.randint(0, N, (1,), generator=g).item())
                step = _coprime_step(N, g)  # gcd(step, N) == 1

                # Global batch per file split evenly across virtual ranks
                # Use disjoint chunks from a single arithmetic progression permutation
                chunk_size = self.per_file_local
                offset = self.virtual_rank * chunk_size  # disjoint across virtual ranks

                # Generate exactly chunk_size unique indices for this virtual rank
                for j in range(chunk_size):
                    ev = (start + (offset + j) * step) % N
                    batch.append(SampleKey(file_idx=fid, event_idx=ev))

            # Light shuffle inside the local batch for extra randomness
            g_batch = torch.Generator()
            g_batch.manual_seed(self.seed * 65537 + self.epoch * 131071 + group_idx * 524287 + self.virtual_rank)
            perm = torch.randperm(len(batch), generator=g_batch).tolist()

            yield [batch[i] for i in perm]