import math
import random
from collections import defaultdict, deque
from typing import Dict, Iterator, List, Sequence

from torch.utils.data import Sampler


def _normalize_ratios(r: Dict[str, float]) -> Dict[str, float]:
    pos = {k: max(0.0, v) for k, v in r.items() if v > 0.0}
    if not pos:
        raise ValueError("All dataset ratios are zero or missing.")
    s = sum(pos.values())
    return {k: pos.get(k, 0.0) / s for k in r}


class BalancedRatioBatchSampler(Sampler[List[int]]):
    """
    Yields batches of indices with per-dataset counts ~ target ratios,
    using online largest-remainder with error feedback to keep every batch close.

    Works for single-GPU or as the 'global' sampler for Distributed use.

    Args:
        dataset_names: a sequence (len==len(dataset)) of dataset ids per example
        ratios: desired share per dataset id (will be normalized over keys present in dataset_names)
        per_batch_size: batch size this sampler emits (use world_size*per_rank_batch for DDP)
        num_batches: number of batches per epoch (defaults to floor(len(dataset)/per_batch_size))
        shuffle_each_epoch: reshuffle per-dataset queues at epoch start
        replacement_after_exhaustion: if a dataset runs out mid-epoch, resample (with replacement) from its pool
        seed: base RNG seed (epoch is added)
        drop_last_incomplete: if True and not enough indices remain to form a full batch, drop the tail
    """

    def __init__(
            self,
            dataset_names: Sequence[str],
            ratios: Dict[str, float],
            per_batch_size: int,
            num_batches: int | None = None,
            shuffle_each_epoch: bool = True,
            replacement_after_exhaustion: bool = True,
            seed: int = 1337,
            drop_last_incomplete: bool = True,
    ):
        super().__init__()

        if per_batch_size <= 0:
            raise ValueError("per_batch_size must be positive")
        self.N = len(dataset_names)
        self.per_batch_size = per_batch_size
        self.shuffle_each_epoch = shuffle_each_epoch
        self.replacement_after_exhaustion = replacement_after_exhaustion
        self.drop_last_incomplete = drop_last_incomplete
        self.seed = seed

        # Build pools of indices per dataset id
        by_ds: Dict[str, List[int]] = defaultdict(list)
        for i, ds in enumerate(dataset_names):
            by_ds[ds].append(i)

        # Restrict ratios to datasets that exist; normalize
        present = {k: ratios.get(k, 0.0) for k in by_ds.keys()}
        self.ds_ratios = _normalize_ratios(present)

        self.ds_ids = sorted(self.ds_ratios.keys())  # deterministic order
        self.pools_all: Dict[str, List[int]] = {k: v for k, v in by_ds.items()}
        self.pools: Dict[str, deque[int]] = {}  # will be populated per epoch

        # Default num_batches
        if num_batches is None:
            max_full = self.N // self.per_batch_size
            num_batches = max_full if drop_last_incomplete else math.ceil(self.N / self.per_batch_size)
        self.num_batches = int(num_batches)

        # For error-feedback rounding across batches
        self._residuals: Dict[str, float] = {k: 0.0 for k in self.ds_ids}
        self._epoch = 0
        self._rng = random.Random(self.seed)

    def set_epoch(self, epoch: int):
        """Call from your training loop (or DistributedSampler hook) to change shuffling each epoch."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.num_batches

    def _reset_epoch_state(self):
        rng = random.Random(self.seed + self._epoch)
        self._residuals = {k: 0.0 for k in self.ds_ids}
        self.pools = {}
        for ds, idxs in self.pools_all.items():
            buf = idxs[:]
            if self.shuffle_each_epoch:
                rng.shuffle(buf)
            self.pools[ds] = deque(buf)

    def _draw_from_pool(self, ds: str, k: int, rng: random.Random) -> List[int]:
        """
        Pop up to k distinct indices from pool; if exhausted and replacement enabled,
        sample with replacement from the dataset's full pool to fill the remainder.
        """
        out: List[int] = []
        pool = self.pools[ds]
        while k > 0 and pool:
            out.append(pool.popleft())
            k -= 1
        if k > 0:
            if not self.replacement_after_exhaustion or not self.pools_all[ds]:
                # Can't fill any further
                return out
            # sample with replacement to meet quota
            src = self.pools_all[ds]
            out.extend(rng.choices(src, k=k))
        return out

    def __iter__(self) -> Iterator[List[int]]:
        self._reset_epoch_state()
        rng = random.Random(self.seed + self._epoch + 17)

        # Precompute ideal per-batch floats for each dataset
        target_per_batch = {ds: self.ds_ratios[ds] * self.per_batch_size for ds in self.ds_ids}

        for b in range(self.num_batches):
            # Online largest-remainder with error feedback:
            # desired = target + residual; take floors, then distribute leftover by largest fractional parts
            desired = {ds: target_per_batch[ds] + self._residuals[ds] for ds in self.ds_ids}
            floors = {ds: int(math.floor(desired[ds])) for ds in self.ds_ids}
            assigned = sum(floors.values())
            leftover = self.per_batch_size - assigned

            if leftover > 0:
                fracs = sorted(
                    [(ds, desired[ds] - floors[ds]) for ds in self.ds_ids],
                    key=lambda x: (x[1], x[0]),
                    reverse=True
                )
                i = 0
                while leftover > 0 and i < len(fracs):
                    ds, _ = fracs[i]
                    floors[ds] += 1
                    leftover -= 1
                    i += 1

            # Update residuals = desired - final
            self._residuals = {ds: desired[ds] - floors[ds] for ds in self.ds_ids}

            # Draw indices according to floors
            batch: List[int] = []
            for ds in self.ds_ids:
                need = floors[ds]
                if need <= 0:
                    continue
                batch.extend(self._draw_from_pool(ds, need, rng))

            # If we couldn't fill due to global exhaustion and drop_last_incomplete, stop
            if len(batch) < self.per_batch_size:
                if self.drop_last_incomplete:
                    break
                # else top up from any available dataset (uniform over all remaining)
                # This fallback will slightly perturb per-batch ratios only at the very end.
                remaining = sum(len(q) for q in self.pools.values())
                if remaining > 0:
                    # Collect leftovers until we reach the target size or pools empty
                    flat = []
                    for ds, q in self.pools.items():
                        flat.extend(list(q))
                    rng.shuffle(flat)
                    need = self.per_batch_size - len(batch)
                    batch.extend(flat[:need])

            # Final sanity (can be < per_batch_size if absolutely no data left)
            yield batch
