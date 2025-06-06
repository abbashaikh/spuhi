import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import matplotlib
matplotlib.use('Agg')

from trajdata import AgentBatch
from traj_pred.utils.sc_sampling import EventSampler

class ProjHead(nn.Module):
    '''
    Nonlinear projection head that maps the extracted motion features to the embedding space
    '''
    def __init__(self, feat_dim, hidden_dim, head_dim):
        super(ProjHead, self).__init__()
        self.head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
            )

    def forward(self, feat):
        """Forward pass of projection head"""
        return self.head(feat)

class EventEncoder(nn.Module):
    '''
    Event encoder that maps a sampled event (location & time) to the embedding space
    '''
    def __init__(self, hidden_dim, head_dim):
        super(EventEncoder, self).__init__()
        self.temporal = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.spatial = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(inplace=True)
            )
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, head_dim)
        )

    def forward(self, state, time):
        """Forward pass of event encoder"""
        emb_state = self.spatial(state)
        emb_time = self.temporal(time)
        return self.encoder(torch.cat([emb_time, emb_state], axis=-1))

class SocialNCE(nn.Module):
    '''
    Social contrastive loss, encourage the extracted motion
    representation to be aware of socially unacceptable events
    '''
    def __init__(
        self,
        feat_dim,
        proj_head_dim,
        event_enc_dim,
        snce_head_dim,
        hyperparams,
        device,
        horizon=3,
        temperature=0.1
    ):
        super(SocialNCE, self).__init__()
        self.hyperparams = hyperparams
        self.device = device
        # encoders
        self.head_projection = ProjHead(
            feat_dim=feat_dim,
            hidden_dim=proj_head_dim,
            head_dim=snce_head_dim
        )
        self.encoder_sample = EventEncoder(
            hidden_dim=event_enc_dim,
            head_dim=snce_head_dim
        )
        # nce
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        # sampling
        self.horizon = horizon
        self.sampler = EventSampler(device=self.device)
    
    def forward(self, enc, batch):
        """Forward pass of Social NCE"""
        return self.loss(enc, batch)
    
    def loss(self, feat, batch: AgentBatch):
        '''
        Social NCE Loss
        '''
        # self._sanity_check(primary_prev, neighbors_prev, primary_next, neighbors_next, sample_pos, sample_neg, mask_valid)
        # pdb.set_trace()

        # Q = max_neighbors*zone_size (to track tensor size)

        (candidate_pos,
         candidate_neg,
         mask_valid_neg) = self.sampler.social_sampling(batch, self.horizon)

        # temporal (normalized time steps)
        time_pos = (torch.ones(candidate_pos.size(0))[:, None]
            * (torch.arange(self.horizon) - (self.horizon-1.0)*(0.5))[None, :]
        ).to(candidate_pos.device) / self.horizon
        time_neg = (torch.ones(candidate_neg.size(0), candidate_neg.size(1))[:, :, None]
            * (torch.arange(self.horizon) - (self.horizon-1.0)*(0.5))[None, None, :]
        ).to(candidate_neg.device) / self.horizon

        # embedding
        emb_obsv = self.head_projection(feat)
        emb_pos = self.encoder_sample(candidate_pos, time_pos[:, :, None])
        emb_neg = self.encoder_sample(candidate_neg, time_neg[:, :, :, None])

        # normalization
        query = nn.functional.normalize(emb_obsv, dim=-1)   # [bs, x_size]
        key_pos = nn.functional.normalize(emb_pos, dim=-1)  # [bs, horizon, head_dim]
        key_neg = nn.functional.normalize(emb_neg, dim=-1)  # [bs, Q, horizon, head_dim]

        # similarity
        sim_pos = (query.unsqueeze(1) * key_pos).sum(dim=-1)                # [bs, horizon]
        sim_neg = (query.unsqueeze(1).unsqueeze(2) * key_neg).sum(dim=-1)   # [bs, Q, horizon]

        # nan post-process: set nan negatives to large negative value
        sim_neg.masked_fill_(~mask_valid_neg[:, :, :self.horizon, 0], -10.0)

        # logits
        sz_neg = sim_neg.size()
        pos_logits = sim_pos.view(sz_neg[0]*sz_neg[2], 1)                           # [bs*horizon, 1]
        neg_logits = sim_neg.permute(0, 2, 1)
        neg_logits = neg_logits.contiguous().view(sz_neg[0]*sz_neg[2], sz_neg[1])   # [bs*horizon, Q]
        logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature      # [bs*horizon, 1+Q]

        # loss
        labels = torch.zeros(logits.size(
            0), dtype=torch.long, device=logits.device)

        loss = self.criterion(logits, labels)

        return loss

    # def _sanity_check(self, primary_prev, neighbors_prev, primary_next, neighbors_next, sample_pos, sample_neg, mask_valid):
    #     '''
    #     Check sampling strategy
    #     '''
    #     for i in range(40):
    #         for k in range(1, self.horizon):
    #             sample_pos_raw = primary_prev[i, -1, :2] + sample_pos[i, k]
    #             sample_neg_raw = primary_prev[i, -1,
    #                                           :2].unsqueeze(0) + sample_neg[i, :, k]
    #             sample_neg_raw = sample_neg_raw[mask_valid[i, :, k].squeeze()]
    #             if len(sample_neg_raw) > 0:
    #                 self._visualize_samples(primary_prev[i, :, :2].cpu().numpy(),
    #                                         neighbors_prev[i, ..., :2].cpu().numpy(),
    #                                         primary_next[i, :, :2].cpu().numpy(),
    #                                         neighbors_next[i, ..., :2].cpu().numpy(),
    #                                         sample_pos_raw.cpu().numpy(), sample_neg_raw.cpu().numpy(),
    #                                         fname='sanity/samples_{:d}_time_{:d}.png'.format(i, k))

    # def _visualize_samples(self, primary_prev_frame, neighbors_prev_frame, primary_next_frame, neighbors_next_frame, sample_pos_frame, sample_neg_frame, fname, window=4.0):

    #     fig = plt.figure(frameon=False)
    #     fig.set_size_inches(8, 6)
    #     ax = fig.add_subplot(1, 1, 1)

    #     ax.plot(primary_prev_frame[:, 0],
    #             primary_prev_frame[:, 1], 'k-o', markersize=4)
    #     ax.plot(primary_next_frame[:, 0], primary_next_frame[:, 1], 'k-.')

    #     for i in range(neighbors_prev_frame.shape[0]):
    #         ax.plot(
    #             neighbors_prev_frame[i, :, 0], neighbors_prev_frame[i, :, 1], 'c-o', markersize=4)
    #         ax.plot(neighbors_next_frame[i, :, 0],
    #                 neighbors_next_frame[i, :, 1], 'c-.')

    #     ax.plot(sample_pos_frame[0], sample_pos_frame[1], 'gs')
    #     ax.plot(sample_neg_frame[:, 0], sample_neg_frame[:, 1], 'rx')

    #     ax.set_xlim(primary_prev_frame[-1, 0] - window, primary_prev_frame[-1, 0] + window)
    #     ax.set_ylim(primary_prev_frame[-1, 1] - window, primary_prev_frame[-1, 1] + window)
    #     ax.set_aspect('equal')
    #     plt.grid()

    #     plt.savefig(fname, bbox_inches='tight', pad_inches=0)
    #     plt.close(fig)
