"""Social contrastive sampling class"""
import torch

from trajdata import AgentBatch

class EventSampler():
    '''
    Different sampling strategies for social contrastive learning
    '''

    def __init__(self, device):
        # fixed param
        self.noise_local = 0.02
        self.min_separation = 0.2                                   # env-dependent parameter, diameter of agents
        self.max_separation = 2.5                                   # env-dependent parameter, diameter of agents
        self.agent_zone = self.min_separation * torch.tensor([
            [1.0, 0.0], [-1.0, 0.0],
            [0.0, 1.0], [0.0, -1.0],
            [0.707, 0.707], [0.707, -0.707],
            [-0.707, 0.707], [-0.707, -0.707]], device=device)      # regional surroundings
        self.device = device

    def _forward_fill_positions(
            self,
            primary_fut: torch.Tensor,
            mask_valid_pos: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized forward-fill of sample_pos along dim=1.

        Assumes: mask_valid_pos[b,0] == True for all b, so there are
                no False slots before the first True.

        Args:
        sample_pos      (B, H, 2)  float tensor
        mask_valid_pos  (B, H)     bool tensor

        Returns:
        (B, H, 2) tensor where each False takes the last True's position
        """
        B, H, D = primary_fut.shape

        # 1) [0,1,2,...,H-1] for each batch
        idx = torch.arange(H, device=mask_valid_pos.device).unsqueeze(0).expand(B, H)

        # 2) zero out invalid slots
        valid_idx = idx * mask_valid_pos.long()

        # 3) running max → at each h, the last index where mask was True
        last_idx, _ = valid_idx.cummax(dim=1)

        # 4) gather those positions
        gather_idx = last_idx.unsqueeze(-1).expand(-1, -1, D)
        return primary_fut.gather(1, gather_idx)

    def _valid_check(self, pos_seed, neg_seed):
        '''
        Check validity of sample seeds, mask out the frames that are invalid due to nan
        '''
        dist = (neg_seed - pos_seed.unsqueeze(1)).norm(dim=-1)
        mask_valid_neg = (dist > self.min_separation) & (dist < self.max_separation)

        dmin = torch.where(
            torch.isnan(dist[mask_valid_neg]),
            torch.full_like(dist[mask_valid_neg], 1000.0),
            dist[mask_valid_neg]
        ).min()
        assert dmin > self.min_separation

        return mask_valid_neg.unsqueeze(-1)

    def social_sampling(self, batch: AgentBatch, horizon):
        '''
        Draw negative samples based on regions of other agents in the future
        '''
        agent_fut_len = batch.agent_fut_len.to(self.device)
        neigh_fut_len = batch.neigh_fut_len.to(self.device)
        valid_fut_len = torch.min(agent_fut_len.max(), neigh_fut_len.max())
        # no samples beyond maximum available primary future
        assert horizon <= valid_fut_len

        # valid positive samples
        ts = torch.arange(horizon, device=agent_fut_len.device)
        mask_valid_pos = ts.unsqueeze(0).lt(agent_fut_len.unsqueeze(1))
        # mask_valid_pos = mask_valid_pos.view(-1)

        # initiate primary and neighbor future tensors
        primary_fut = batch.agent_fut[..., :horizon, :2]
        primary_fut = self._forward_fill_positions(primary_fut, mask_valid_pos)
        neigh_fut = batch.neigh_fut[..., :horizon, :2]
        # positive
        sample_pos = primary_fut + torch.rand(
                primary_fut.size(),
                device=self.device
            ).sub(0.5) * self.noise_local # shape -> [bs, max_fut_len, 2]
        
        # neighbor territory
        sample_neg = neigh_fut.unsqueeze(2) + self.agent_zone[None, None, :, None, :]
        sample_neg = sample_neg.view(
            sample_neg.size(0),
            sample_neg.size(1)*sample_neg.size(2),
            sample_neg.size(3),
            2
        )
        sample_neg += torch.rand(
            sample_neg.size(),
            device=self.device
        ).sub(0.5) * self.noise_local
        # sample_neg.shape -> [bs, max_neighbors*zone_size, max_fut_len, 2]

        # check for valid negative samples
        mask_valid_neg = self._valid_check(sample_pos, sample_neg)
        # mask_valid.shape -> [bs, max_neighbors*zone_size, max_fut_len, 1]
        sample_neg.masked_fill_(~mask_valid_neg, 0.0)

        return sample_pos, sample_neg, mask_valid_neg
