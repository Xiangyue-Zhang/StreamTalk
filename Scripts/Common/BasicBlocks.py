import torch.nn as nn
import torch

class DropPathFast(nn.Module):
    """内存优化版本，避免显式创建mask张量"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        mask = torch.rand((x.size(0), 1,  1), device=x.device) < keep_prob
        return x * mask / keep_prob


# 2
class ResBlock1D_2(nn.Module):
    def __init__(self, input_channels, output_channels=None, kernel_size=3, time_emb_dim=None):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size 必须是奇数"
        output_channels = output_channels or input_channels
        
        # 时间嵌入投影
        if time_emb_dim is not None:
            self.time_mlp = nn.Linear(time_emb_dim, input_channels)
        else:
            self.time_mlp = None

        # 主路径
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(output_channels)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(output_channels)

        # 残差捷径
        self.short_cut = (
            nn.Identity() 
            if input_channels == output_channels 
            else nn.Conv1d(input_channels, output_channels, 1)
        )
        self.scale = nn.Parameter(torch.FloatTensor([1e-6]))
        self.droppath = DropPathFast(0.1)

    def forward(self, x, time_emb=None):
        x0 = x  # 残差连接

        # 主路径处理
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)

        # 时间嵌入融合
        if time_emb is not None and self.time_mlp is not None:
            time_emb = self.time_mlp(time_emb).unsqueeze(-1)  # (B, C, 1)
            x = x + time_emb

        x = self.conv2(x)
        x = self.bn2(x)

        # 残差连接
        x = self.droppath(x * self.scale) + self.short_cut(x0)
        x = self.act(x)  # 最终激活

        return x
 
#1
class ResBlock1D(nn.Module):
    def __init__(self, input_channels, output_channels=None, kernel_size=3):
        super().__init__()
        assert kernel_size % 2 == 1, "only support odd kernel_size"
        output_channels = output_channels if output_channels is not None else input_channels
        self.bn1 = nn.BatchNorm1d(input_channels)
        self.bn2 = nn.BatchNorm1d(output_channels)
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size, padding=kernel_size//2)

        self.act = nn.SiLU()

        self.short_cut = nn.Identity() if input_channels == output_channels else nn.Conv1d(input_channels, output_channels, 1)
        
    def forward(self, x, time_emb=None):
        x0 = x
        x = self.bn1(x)
        x = self.act(x)
        if time_emb is not None:
            x = x + time_emb[:,:,None]
        x = self.conv1(x)
        
        x = self.bn2(x)
        x = self.act(x)
        x = self.conv2(x)
        x = x + self.short_cut(x0)
        return x
    


class Encoder(nn.Module): 
    def __init__(self, 
                 in_channels,
                 model_channels=[256, 512, 512],
                 down_block_ids = [0, 1],
                 num_res_per_blocks=2,
                 down_sample="pool",
                 final_output_channels=None,
                 kernel_size = [3, 3, 3],
                 block_type="res1"):
        super().__init__()
        assert len(model_channels) == len(kernel_size)
        self.in_prj = nn.Conv1d(in_channels, model_channels[0], 1)
        pre_ch = model_channels[0]

        BLOCK_MODULE = {
            "res1": ResBlock1D,
            "res2": ResBlock1D_2,
        }[block_type]
        
        blks = []
        for i in range(len(model_channels)):
            ch = model_channels[i]
            blks.append(BLOCK_MODULE(pre_ch, ch, kernel_size=kernel_size[i]))
            blks = blks + [BLOCK_MODULE(ch, ch, kernel_size=kernel_size[i]) for _ in range(num_res_per_blocks-1)]
            if i in down_block_ids:
                if down_sample != "conv":
                    blks.append(nn.MaxPool1d(2, 2))
                else:
                    blks.append(nn.Conv1d(ch, ch, kernel_size=2, stride=2))
            pre_ch = ch
        self.blks = nn.Sequential(*blks)
        if final_output_channels is None:
            final_output_channels = model_channels[-1]
        self.prj_out = nn.Conv1d(
            model_channels[-1], final_output_channels, kernel_size=1)

    def forward(self, x):
        x = self.in_prj(x)
        x = self.blks(x)
        x = self.prj_out(x)
        return x
    
    

class Decoder(nn.Module):
    def __init__(self, 
                 in_channels,
                 model_channels=[512,512, 256],
                 up_block_ids = [0,1],
                 num_res_per_block=2,
                 up_sample="upsample",
                 final_output_channels = None,
                 kernel_size = [3, 3, 3],
                 block_type="res1"
                 ):
        super().__init__()
        assert len(kernel_size) == len(model_channels)
        self.prj_in = nn.Conv1d(in_channels, model_channels[0], 1)
        pre_ch = model_channels[0]
        blks = []
        BLOCK_MODULE = {
            "res1": ResBlock1D,
            "res2": ResBlock1D_2,
        }[block_type]
        for i in range(len(model_channels)):
            ch = model_channels[i]
            blks.append(BLOCK_MODULE(pre_ch, ch, kernel_size=kernel_size[i]))
            blks = blks + [BLOCK_MODULE(ch, ch, kernel_size=kernel_size[i]) for _ in range(num_res_per_block-1)]
            if i in up_block_ids:
                if up_sample != "conv":
                    blks.append(nn.Upsample(scale_factor=2, mode='linear', align_corners=True))
                else:
                    blks.append(nn.ConvTranspose1d(
                        ch, ch, kernel_size=2, stride=2, dilation=1))
            pre_ch = ch

        self.blks = nn.Sequential(*blks)
        if final_output_channels is None:
            final_output_channels = model_channels[-1]
        self.out_prj = nn.Conv1d(model_channels[-1], final_output_channels, 1)

    def forward(self, x):
        x = self.prj_in(x)
        x = self.blks(x)
        x = self.out_prj(x)
        return x
    
if __name__ == "__main__":
    x = torch.rand(2,32,64)
    E = Encoder(in_channels=32, model_channels=[256,512,512], down_block_ids=[], num_res_per_blocks=2, kernel_size=[5,5,5], block_type="res2")
    D = Decoder(in_channels=512, model_channels=[256,512,256], up_block_ids=[0,1], num_res_per_block=2, kernel_size=[5,5,5], block_type="res2")
    
    r = E(x)
    print(r.shape)
    # y = D(r)
    # print(y.shape)