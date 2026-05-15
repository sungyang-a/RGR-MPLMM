
import torch
import torch.nn.functional as F
from torch import nn

from modules.transformer import TransformerEncoder


_MISSING_TABLE = torch.tensor(
    [
        [1.0, 0.0, 0.0],  # 0: l 缺
        [0.0, 1.0, 0.0],  # 1: a 缺
        [0.0, 0.0, 1.0],  # 2: v 缺
        [1.0, 1.0, 0.0],  # 3: l + a 缺
        [1.0, 0.0, 1.0],  # 4: l + v 缺
        [0.0, 1.0, 1.0],  # 5: a + v 缺
        [0.0, 0.0, 0.0],  # 6: 全模态
    ],
    dtype=torch.float32,
)


def _build_missing_mask(missing_mod, device):
   
    if not torch.is_tensor(missing_mod):
        missing_mod = torch.as_tensor(missing_mod, dtype=torch.long, device=device)
    elif missing_mod.dtype != torch.long:
        missing_mod = missing_mod.long()
    table = _MISSING_TABLE.to(device=device, dtype=torch.float32)
    return table.index_select(0, missing_mod.to(device))  # [B, 3]



class _PlainConcat(nn.Module):
    def forward(self, h_l_n, h_a_n, h_v_n, miss_mask):
        last_hs = torch.cat([h_l_n, h_a_n, h_v_n], dim=-1)
        B = h_l_n.size(0)
        w = torch.full((B, 3), 1.0 / 3.0, device=h_l_n.device, dtype=h_l_n.dtype)
        return last_hs, w



class ReliabilityGate(nn.Module):
   

    def __init__(self, dim, hidden_ratio=2):
        super().__init__()
        hidden = max(dim // hidden_ratio, 4)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h):
        return self.fc2(F.gelu(self.fc1(h))).squeeze(-1)



class GatedFusion(nn.Module):
   

    def __init__(self, dim):
        super().__init__()
        self.gate_l = ReliabilityGate(dim)
        self.gate_a = ReliabilityGate(dim)
        self.gate_v = ReliabilityGate(dim)

        self.miss_penalty_logit = nn.Parameter(torch.tensor(-2.0))
 
        self.inject_l = nn.Linear(dim, dim)
        self.inject_a = nn.Linear(dim, dim)
        self.inject_v = nn.Linear(dim, dim)
        for m in (self.inject_l, self.inject_a, self.inject_v):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
  
        self.inject_beta = nn.Parameter(torch.tensor(0.5))

    def forward(self, h_l_n, h_a_n, h_v_n, miss_mask):
        # 打分
        s_l = self.gate_l(h_l_n)
        s_a = self.gate_a(h_a_n)
        s_v = self.gate_v(h_v_n)


        penalty = F.softplus(self.miss_penalty_logit)  # ≥ 0,标量
        s_l = s_l - miss_mask[:, 0] * penalty
        s_a = s_a - miss_mask[:, 1] * penalty
        s_v = s_v - miss_mask[:, 2] * penalty

        scores = torch.stack([s_l, s_a, s_v], dim=-1)  # [B, 3]
        weights = F.softmax(scores, dim=-1)             # [B, 3]
        w_l = weights[:, 0:1]
        w_a = weights[:, 1:2]
        w_v = weights[:, 2:3]


        h_w = w_l * h_l_n + w_a * h_a_n + w_v * h_v_n   # [B, dim]


        delta = torch.cat(
            [self.inject_l(h_w), self.inject_a(h_w), self.inject_v(h_w)], dim=-1
        )

        last_hs = torch.cat([h_l_n, h_a_n, h_v_n], dim=-1)
        last_hs = last_hs + self.inject_beta * delta
        return last_hs, weights



class MULTModel(nn.Module):
    def __init__(self, hyp_params):
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = (
            hyp_params.orig_d_l,
            hyp_params.orig_d_a,
            hyp_params.orig_d_v,
        )
        self.d_l, self.d_a, self.d_v = (
            hyp_params.proj_dim,
            hyp_params.proj_dim,
            hyp_params.proj_dim,
        )
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.prompt_length = hyp_params.prompt_length
        self.prompt_dim = hyp_params.prompt_dim
        self.llen, self.alen, self.vlen = hyp_params.seq_len
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        output_dim = hyp_params.output_dim

        # 1. 时序卷积投影
        self.proj_l = nn.Conv1d(
            self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False
        )
        self.proj_a = nn.Conv1d(
            self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False
        )
        self.proj_v = nn.Conv1d(
            self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False
        )

        # 2. 跨模态注意力
        self.trans_l_with_a = self.get_network(self_type="la")
        self.trans_l_with_v = self.get_network(self_type="lv")

        self.trans_a_with_l = self.get_network(self_type="al")
        self.trans_a_with_v = self.get_network(self_type="av")

        self.trans_v_with_l = self.get_network(self_type="vl")
        self.trans_v_with_a = self.get_network(self_type="va")

        # 3. 自注意力 memory
        self.trans_l_mem = self.get_network(self_type="l_mem", layers=3)
        self.trans_a_mem = self.get_network(self_type="a_mem", layers=3)
        self.trans_v_mem = self.get_network(self_type="v_mem", layers=3)

        # === 改进 M1: 模态级 LayerNorm(预训练时也启用,便于权重迁移) ===
        self.mod_ln_l = nn.LayerNorm(2 * self.d_l)
        self.mod_ln_a = nn.LayerNorm(2 * self.d_a)
        self.mod_ln_v = nn.LayerNorm(2 * self.d_v)

        # 投影分类头
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type="l", layers=-1):
        if self_type in ["l", "al", "vl"]:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ["a", "la", "va"]:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ["v", "lv", "av"]:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == "l_mem":
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == "a_mem":
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == "v_mem":
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def forward(self, x_l, x_a, x_v, missing_mod=None, return_features=False):
        x_l = F.dropout(
            x_l.transpose(1, 2), p=self.embed_dropout, training=self.training
        )
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        proj_x_a = proj_x_a.permute(2, 0, 1)
        proj_x_v = proj_x_v.permute(2, 0, 1)
        proj_x_l = proj_x_l.permute(2, 0, 1)

        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = self.trans_l_mem(h_ls)
        if isinstance(h_ls, tuple):
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]

        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = self.trans_a_mem(h_as)
        if isinstance(h_as, tuple):
            h_as = h_as[0]
        last_h_a = h_as[-1]

        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = self.trans_v_mem(h_vs)
        if isinstance(h_vs, tuple):
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]

        # 改进 M1: 各模态独立 LayerNorm
        h_l_n = self.mod_ln_l(last_h_l)
        h_a_n = self.mod_ln_a(last_h_a)
        h_v_n = self.mod_ln_v(last_h_v)

        last_hs = torch.cat([h_l_n, h_a_n, h_v_n], dim=1)

        last_hs_proj = self.proj2(
            F.dropout(
                F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training
            )
        )
        last_hs_proj = last_hs_proj + last_hs

        output = self.out_layer(last_hs_proj)
        if return_features:
            return output, {"h_l": last_h_l, "h_a": last_h_a, "h_v": last_h_v}
        return output


class PromptModel(nn.Module):
    def __init__(self, hyp_params):
        super(PromptModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = (
            hyp_params.orig_d_l,
            hyp_params.orig_d_a,
            hyp_params.orig_d_v,
        )
        self.d_l, self.d_a, self.d_v = (
            hyp_params.proj_dim,
            hyp_params.proj_dim,
            hyp_params.proj_dim,
        )
        self.num_heads = hyp_params.num_heads
        self.layers = hyp_params.layers
        self.attn_dropout = hyp_params.attn_dropout
        self.attn_dropout_a = hyp_params.attn_dropout_a
        self.attn_dropout_v = hyp_params.attn_dropout_v
        self.relu_dropout = hyp_params.relu_dropout
        self.res_dropout = hyp_params.res_dropout
        self.out_dropout = hyp_params.out_dropout
        self.embed_dropout = hyp_params.embed_dropout
        self.attn_mask = hyp_params.attn_mask
        self.prompt_length = hyp_params.prompt_length
        self.prompt_dim = hyp_params.prompt_dim
        self.llen, self.alen, self.vlen = hyp_params.seq_len
        combined_dim = 2 * (self.d_l + self.d_a + self.d_v)
        output_dim = hyp_params.output_dim

        # 生成式提示
        generative_prompt = torch.zeros(3, self.prompt_dim, self.prompt_length)
        self.generative_prompt = nn.Parameter(generative_prompt)

        # 跨模态生成 MLP(与基线一致)
        self.l2a = MLPLayer(self.orig_d_l, self.prompt_dim)
        self.l2v = MLPLayer(self.orig_d_l, self.prompt_dim)
        self.v2a = MLPLayer(self.orig_d_v, self.prompt_dim)
        self.v2l = MLPLayer(self.orig_d_v, self.prompt_dim)
        self.a2v = MLPLayer(self.orig_d_a, self.prompt_dim)
        self.a2l = MLPLayer(self.orig_d_a, self.prompt_dim)

        self.l_ap = MLPLayer(self.prompt_length + self.alen, self.llen, True)
        self.l_vp = MLPLayer(self.prompt_length + self.vlen, self.llen, True)
        self.l_avp = MLPLayer(
            self.prompt_length + self.alen + self.vlen, self.llen, True
        )

        self.a_lp = MLPLayer(self.prompt_length + self.llen, self.alen, True)
        self.a_vp = MLPLayer(self.prompt_length + self.vlen, self.alen, True)
        self.a_lvp = MLPLayer(
            self.prompt_length + self.llen + self.vlen, self.alen, True
        )

        self.v_ap = MLPLayer(self.prompt_length + self.alen, self.vlen, True)
        self.v_lp = MLPLayer(self.prompt_length + self.llen, self.vlen, True)
        self.v_alp = MLPLayer(
            self.prompt_length + self.alen + self.llen, self.vlen, True
        )

        # 时序卷积投影
        self.proj_l = nn.Conv1d(
            self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False
        )
        self.proj_a = nn.Conv1d(
            self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False
        )
        self.proj_v = nn.Conv1d(
            self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False
        )

        # modality-signal prompts
        self.promptl_m = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.prompta_m = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.promptv_m = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))
        self.promptl_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.llen))
        self.prompta_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.alen))
        self.promptv_nm = nn.Parameter(torch.zeros(self.prompt_dim, self.vlen))

        # missing-type prompts
        self.missing_type_prompt = nn.Parameter(
            torch.zeros(3, self.prompt_length, self.prompt_dim)
        )
        self.m_a = nn.Parameter(torch.zeros(self.alen, 2 * self.prompt_dim))
        self.m_v = nn.Parameter(torch.zeros(self.vlen, 2 * self.prompt_dim))
        self.m_l = nn.Parameter(torch.zeros(self.llen, 2 * self.prompt_dim))

        # 跨模态注意力
        self.trans_l_with_a = self.get_network(self_type="la")
        self.trans_l_with_v = self.get_network(self_type="lv")

        self.trans_a_with_l = self.get_network(self_type="al")
        self.trans_a_with_v = self.get_network(self_type="av")

        self.trans_v_with_l = self.get_network(self_type="vl")
        self.trans_v_with_a = self.get_network(self_type="va")


        self.trans_l_mem = self.get_network(self_type="l_mem", layers=3)
        self.trans_a_mem = self.get_network(self_type="a_mem", layers=3)
        self.trans_v_mem = self.get_network(self_type="v_mem", layers=3)


        self.mod_ln_l = nn.LayerNorm(2 * self.d_l)
        self.mod_ln_a = nn.LayerNorm(2 * self.d_a)
        self.mod_ln_v = nn.LayerNorm(2 * self.d_v)

  
        self.fusion_gate = GatedFusion(2 * self.d_l)

    
        self.proj1 = nn.Linear(combined_dim, combined_dim)
        self.proj2 = nn.Linear(combined_dim, combined_dim)
        self.out_layer = nn.Linear(combined_dim, output_dim)

    def get_network(self, self_type="l", layers=-1):
        if self_type in ["l", "al", "vl"]:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ["a", "la", "va"]:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ["v", "lv", "av"]:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        elif self_type == "l_mem":
            embed_dim, attn_dropout = 2 * self.d_l, self.attn_dropout
        elif self_type == "a_mem":
            embed_dim, attn_dropout = 2 * self.d_a, self.attn_dropout
        elif self_type == "v_mem":
            embed_dim, attn_dropout = 2 * self.d_v, self.attn_dropout
        else:
            raise ValueError("Unknown network type")

        return TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=self.num_heads,
            layers=max(self.layers, layers),
            attn_dropout=attn_dropout,
            relu_dropout=self.relu_dropout,
            res_dropout=self.res_dropout,
            embed_dropout=self.embed_dropout,
            attn_mask=self.attn_mask,
        )

    def get_complete_data(self, x_l, x_a, x_v, missing_mode):
        x_l, x_a, x_v = x_l.unsqueeze(dim=0), x_a.unsqueeze(dim=0), x_v.unsqueeze(dim=0)
        if missing_mode == 0:
            x_l = torch.cat(
                [self.generative_prompt[0, :, :], self.a2l(x_a)[0], self.v2l(x_v)[0]],
                dim=1,
            ).unsqueeze(dim=0)
            x_l = self.l_avp(x_l.transpose(1, 2)).transpose(1, 2) + self.promptl_m
            x_a = self.proj_a(x_a) + self.prompta_nm
            x_v = self.proj_v(x_v) + self.promptv_nm
        elif missing_mode == 1:
            x_a = torch.cat(
                [self.generative_prompt[1, :, :], self.l2a(x_l)[0], self.v2a(x_v)[0]],
                dim=1,
            ).unsqueeze(dim=0)
            x_a = self.a_lvp(x_a.transpose(1, 2)).transpose(1, 2) + self.prompta_m
            x_v = self.proj_v(x_v) + self.promptv_nm
            x_l = self.proj_l(x_l) + self.promptl_nm
        elif missing_mode == 2:
            x_v = torch.cat(
                [self.generative_prompt[2, :, :], self.l2v(x_l)[0], self.a2v(x_a)[0]],
                dim=1,
            ).unsqueeze(dim=0)
            x_v = self.v_alp(x_v.transpose(1, 2)).transpose(1, 2) + self.promptv_m
            x_l = self.proj_l(x_l) + self.promptl_nm
            x_a = self.proj_a(x_a) + self.prompta_nm
        elif missing_mode == 3:
            x_l = torch.cat(
                [self.generative_prompt[0, :, :], self.v2l(x_v)[0]], dim=1
            ).unsqueeze(dim=0)
            x_a = torch.cat(
                [self.generative_prompt[1, :, :], self.v2a(x_v)[0]], dim=1
            ).unsqueeze(dim=0)
            x_l = self.l_vp(x_l.transpose(1, 2)).transpose(1, 2) + self.promptl_m
            x_a = self.a_vp(x_a.transpose(1, 2)).transpose(1, 2) + self.prompta_m
            x_v = self.proj_v(x_v) + self.promptv_nm
        elif missing_mode == 4:
            x_l = torch.cat(
                [self.generative_prompt[0, :, :], self.a2l(x_a)[0]], dim=1
            ).unsqueeze(dim=0)
            x_v = torch.cat(
                [self.generative_prompt[2, :, :], self.a2v(x_a)[0]], dim=1
            ).unsqueeze(dim=0)
            x_l = self.l_ap(x_l.transpose(1, 2)).transpose(1, 2) + self.promptl_m
            x_v = self.v_ap(x_v.transpose(1, 2)).transpose(1, 2) + self.promptv_m
            x_a = self.proj_a(x_a) + self.prompta_nm
        elif missing_mode == 5:
            x_a = torch.cat(
                [self.generative_prompt[1, :, :], self.l2a(x_l)[0]], dim=1
            ).unsqueeze(dim=0)
            x_v = torch.cat(
                [self.generative_prompt[2, :, :], self.l2v(x_l)[0]], dim=1
            ).unsqueeze(dim=0)
            x_a = self.a_lp(x_a.transpose(1, 2)).transpose(1, 2) + self.prompta_m
            x_v = self.v_lp(x_v.transpose(1, 2)).transpose(1, 2) + self.promptv_m
            x_l = self.proj_l(x_l) + self.promptl_nm
        else:
            x_a = self.proj_a(x_a) + self.prompta_nm
            x_l = self.proj_l(x_l) + self.promptl_nm
            x_v = self.proj_v(x_v) + self.promptv_nm

        return x_l, x_a, x_v

    def get_proj_matrix(self):
        a_v_l = (
            self.prompta_nm @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(dim=0)
        am_v_l = (
            self.prompta_m @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(dim=0)
        a_vm_l = (
            self.prompta_nm @ self.m_a
            + self.promptv_m @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(dim=0)
        a_v_lm = (
            self.prompta_nm @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_m @ self.m_l
        ).unsqueeze(dim=0)
        am_vm_l = (
            self.prompta_m @ self.m_a
            + self.promptv_m @ self.m_v
            + self.promptl_nm @ self.m_l
        ).unsqueeze(dim=0)
        am_v_lm = (
            self.prompta_m @ self.m_a
            + self.promptv_nm @ self.m_v
            + self.promptl_m @ self.m_l
        ).unsqueeze(dim=0)
        a_vm_lm = (
            self.prompta_nm @ self.m_a
            + self.promptv_m @ self.m_v
            + self.promptl_m @ self.m_l
        ).unsqueeze(dim=0)
        self.mp = torch.cat(
            [a_v_lm, am_v_l, a_vm_l, am_v_lm, a_vm_lm, am_vm_l, a_v_l], dim=0
        )

    def forward(self, x_l, x_a, x_v, missing_mod, return_features=False):
        # ------- 1. 数据补全 + missing-signal/missing-type prompts ----
        x_l = F.dropout(
            x_l.transpose(1, 2), p=self.embed_dropout, training=self.training
        )
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        xx_l, xx_a, xx_v = None, None, None
        for idx in range(len(x_l)):
            x_l_temp, x_a_temp, x_v_temp = self.get_complete_data(
                x_l[idx], x_a[idx], x_v[idx], missing_mod[idx]
            )
            if xx_l is None:
                xx_l = x_l_temp
                xx_a = x_a_temp
                xx_v = x_v_temp
            else:
                xx_l = torch.cat([xx_l, x_l_temp], dim=0)
                xx_a = torch.cat([xx_a, x_a_temp], dim=0)
                xx_v = torch.cat([xx_v, x_v_temp], dim=0)

        proj_x_a = xx_a.permute(2, 0, 1)
        proj_x_v = xx_v.permute(2, 0, 1)
        proj_x_l = xx_l.permute(2, 0, 1)

        self.get_proj_matrix()
        batch_prompt = None
        for idx in range(len(x_l)):
            if batch_prompt is None:
                batch_prompt = torch.matmul(
                    self.missing_type_prompt, self.mp[missing_mod[idx]]
                ).unsqueeze(dim=0)
            else:
                batch_prompt = torch.cat(
                    [
                        batch_prompt,
                        torch.matmul(
                            self.missing_type_prompt, self.mp[missing_mod[idx]]
                        ).unsqueeze(dim=0),
                    ],
                    dim=0,
                )

        batch_prompt = batch_prompt.transpose(0, 1)

        h_l_with_as = self.trans_l_with_a(proj_x_l, proj_x_a, proj_x_a)
        h_l_with_vs = self.trans_l_with_v(proj_x_l, proj_x_v, proj_x_v)
        h_ls = torch.cat([h_l_with_as, h_l_with_vs], dim=2)
        h_ls = torch.cat([h_ls, batch_prompt[0].transpose(0, 1)], dim=0)
        h_ls = self.trans_l_mem(h_ls)
        if isinstance(h_ls, tuple):
            h_ls = h_ls[0]
        last_h_l = h_ls[-1]

        h_a_with_ls = self.trans_a_with_l(proj_x_a, proj_x_l, proj_x_l)
        h_a_with_vs = self.trans_a_with_v(proj_x_a, proj_x_v, proj_x_v)
        h_as = torch.cat([h_a_with_ls, h_a_with_vs], dim=2)
        h_as = torch.cat([h_as, batch_prompt[1].transpose(0, 1)], dim=0)
        h_as = self.trans_a_mem(h_as)
        if isinstance(h_as, tuple):
            h_as = h_as[0]
        last_h_a = h_as[-1]

        h_v_with_ls = self.trans_v_with_l(proj_x_v, proj_x_l, proj_x_l)
        h_v_with_as = self.trans_v_with_a(proj_x_v, proj_x_a, proj_x_a)
        h_vs = torch.cat([h_v_with_ls, h_v_with_as], dim=2)
        h_vs = torch.cat([h_vs, batch_prompt[2].transpose(0, 1)], dim=0)
        h_vs = self.trans_v_mem(h_vs)
        if isinstance(h_vs, tuple):
            h_vs = h_vs[0]
        last_h_v = h_vs[-1]


        h_l_n = self.mod_ln_l(last_h_l)
        h_a_n = self.mod_ln_a(last_h_a)
        h_v_n = self.mod_ln_v(last_h_v)


        miss_mask = _build_missing_mask(missing_mod, last_h_l.device)
        last_hs, fusion_weights = self.fusion_gate(h_l_n, h_a_n, h_v_n, miss_mask)


        last_hs_proj = self.proj2(
            F.dropout(
                F.relu(self.proj1(last_hs)), p=self.out_dropout, training=self.training
            )
        )
        last_hs_proj = last_hs_proj + last_hs

        output = self.out_layer(last_hs_proj)
        if return_features:
            return output, {
                "h_l": last_h_l,
                "h_a": last_h_a,
                "h_v": last_h_v,
                "weights": fusion_weights,
            }
        return output



class MLPLayer(nn.Module):
    def __init__(self, dim, embed_dim, is_Fusion=False):
        super().__init__()
        if is_Fusion:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        else:
            self.conv = nn.Conv1d(dim, embed_dim, kernel_size=1, padding=0)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))
