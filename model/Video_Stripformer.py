import torch
import torch.nn as nn
import math


class Embeddings(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3):
        super(Embeddings, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)
        self.en_layer1_1 = nn.Sequential(
            nn.Conv2d(3, dim_1, kernel_size=3, padding=1),
            self.activation,
        )
        self.en_layer1_2 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.en_layer1_3 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.en_layer1_4 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))

        self.en_layer2_1 = nn.Sequential(
            nn.Conv2d(dim_1, dim_2, kernel_size=3, stride=2, padding=1),
            self.activation,
        )
        self.en_layer2_2 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.en_layer2_3 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))
        self.en_layer2_4 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1))

        self.en_layer3_1 = nn.Sequential(
            nn.Conv2d(dim_2, dim_3, kernel_size=3, stride=2, padding=1),
            self.activation,
        )


    def forward(self, x):
        b, f, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        hx = self.en_layer1_1(x)
        hx = self.activation(self.en_layer1_2(hx) + hx)
        hx = self.activation(self.en_layer1_3(hx) + hx)
        hx = self.activation(self.en_layer1_4(hx) + hx)
        _, c, h, w = hx.size()

        hx = self.en_layer2_1(hx)
        hx = self.activation(self.en_layer2_2(hx) + hx)
        hx = self.activation(self.en_layer2_3(hx) + hx)
        hx = self.activation(self.en_layer2_4(hx) + hx)
        _, c, h, w = hx.size()

        hx = self.en_layer3_1(hx)
        _, c, h, w = hx.shape
        hx = hx.view(b, f, c, h, w)

        return hx


class Embeddings_output(nn.Module):
    def __init__(self, dim_1, dim_2, dim_3):
        super(Embeddings_output, self).__init__()

        self.activation = nn.LeakyReLU(0.2, True)

        self.de_layer3_1 = nn.Sequential(
            nn.ConvTranspose2d(dim_3, dim_2, kernel_size=4, stride=2, padding=1),
            self.activation,)

        self.de_layer2_1 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),)
        self.de_layer2_2 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),)
        self.de_layer2_3 = nn.Sequential(
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_2, dim_2, kernel_size=3, padding=1),)
        self.de_layer2_4 = nn.Sequential(
            nn.ConvTranspose2d(dim_2, dim_1, kernel_size=4, stride=2, padding=1),
            self.activation,)

        self.de_layer1_1 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.de_layer1_2 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.de_layer1_3 = nn.Sequential(
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1),
            self.activation,
            nn.Conv2d(dim_1, dim_1, kernel_size=3, padding=1))
        self.de_layer1_4 = nn.Sequential(
            nn.Conv2d(dim_1, 3, kernel_size=3, padding=1),
            self.activation)

    def forward(self, x):
        b, f, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        hx = self.de_layer3_1(x)

        _, c, h, w = hx.size()
        hx = self.activation(self.de_layer2_1(hx) + hx)
        hx = self.activation(self.de_layer2_2(hx) + hx)
        hx = self.activation(self.de_layer2_3(hx) + hx)
        hx = self.de_layer2_4(hx)

        _, c, h, w = hx.size()
        hx = self.activation(self.de_layer1_1(hx) + hx)
        hx = self.activation(self.de_layer1_2(hx) + hx)
        hx = self.activation(self.de_layer1_3(hx) + hx)
        hx = self.de_layer1_4(hx)

        _, c, h, w = hx.shape
        hx = hx.view(b, f, c, h, w)

        return hx

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query_layer, key_layer, value_layer):
        # query_layer: B, head, N, D
        C = query_layer.size()[-1]
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))  # B, Head, N, N
        attention_scores = attention_scores / math.sqrt(C)
        attention_probs = self.softmax(attention_scores)
        #logging.debug((attention_probs.shape, 'attention_probs'))
        attention_out = torch.matmul(attention_probs, value_layer)  # B, Head, N, C
        return attention_out


class Mlp(nn.Module):
    def __init__(self, hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, 4 * hidden_size)
        self.fc2 = nn.Linear(4 * hidden_size, hidden_size)
        self.act_fn = torch.nn.functional.gelu

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.fc2(x)
        return x


class Spatial_Inter_SA(nn.Module):
    def __init__(self, dim, head_num):
        super(Spatial_Inter_SA, self).__init__()
        self.dim_prime = dim // 2
        self.head_num = head_num
        self.head_dim = self.dim_prime // self.head_num
        self.attention_norm = nn.LayerNorm(dim)
        self.conv_h = nn.Conv2d(self.dim_prime, 3 * self.dim_prime, kernel_size=1, padding=0)
        self.conv_v = nn.Conv2d(self.dim_prime, 3 * self.dim_prime, kernel_size=1, padding=0)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.attn = Attention()

    def forward(self, x):
        b, f, c, h, w = x.size()
        x = x.view(-1, c, h, w)

        hx = x
        x = x.view(-1, c, h * w).permute(0, 2, 1).contiguous()  # b*f, h*w, c
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()  # b*f, c, h*w
        x = x.view(-1, c, h, w)  # b*f, c, h, w
        x_input = torch.chunk(x, 2, dim=1)  # 2, b*f, c', h, w

        feature_h = self.conv_h(x_input[0])  # b*f, 3c', h, w
        feature_h = feature_h.view(-1, 3, self.dim_prime, h, w)  # b*f, 3, c', h, w (q, k, v)
        feature_h = feature_h.view(-1, 3, self.head_num, self.head_dim, h, w)  # b*f, 3, head, c'/head, h, w
        feature_h = feature_h.permute(0, 1, 2, 4, 5, 3).contiguous()  # b*f, 3, head, h, w, c'/head
        feature_h = feature_h.view(-1, 3, self.head_num, h, w * self.head_dim)  # b*f, 3, head, h, w*c'/head
        query_h, key_h, value_h = feature_h[:, 0], feature_h[:, 1], feature_h[:, 2]  # b*f, head, h, w*c'/head

        feature_v = self.conv_v(x_input[1])  # b*f, 3c', h, w
        feature_v = feature_v.view(-1, 3, self.dim_prime, h, w)  # b*f, 3, c', h, w (q, k, v)
        feature_v = feature_v.view(-1, 3, self.head_num, self.head_dim, h, w)  # b*f, 3, head, c'/head, h, w
        feature_v = feature_v.permute(0, 1, 2, 5, 4, 3).contiguous()  # b*f, 3, head, w, h, c'/head
        feature_v = feature_v.view(-1, 3, self.head_num, w, h * self.head_dim)  # b*f, 3, head, w, h*c'/head
        query_v, key_v, value_v = feature_v[:, 0], feature_v[:, 1], feature_v[:, 2]  # b*f, head, w, h*c'/head

        if h == w:
            query = torch.cat((query_h, query_v), dim=0)  # B, head, N, D (B=2*b*f, head, N=h=w, D=h*c'/head=w*c'/head)
            key = torch.cat((key_h, key_v), dim=0)  # B, head, N, D
            value = torch.cat((value_h, value_v), dim=0)  # B, head, N, D
            attention_output = self.attn(query, key, value)  # B, head, N, D
            attention_output = torch.chunk(attention_output, 2, dim=0)  # 2, b*f, head, N, D
            attention_output_h = attention_output[0]  # b*f, head, h, w*c'/head
            attention_output_v = attention_output[1]  # b*f, head, w, h*c'/head

            attention_output_h = attention_output_h.view(-1, self.head_num, h, w, self.head_dim)  # b*f, head, h, w, c'/head
            attention_output_h = attention_output_h.permute(0, 1, 4, 2, 3).contiguous()  # b*f, head, c'/head, h, w
            attention_output_h = attention_output_h.view(-1, self.dim_prime, h, w)  # b*f, c', h, w

            attention_output_v = attention_output_v.view(-1, self.head_num, w, h, self.head_dim)  # b*f, head, w, h, c'/head
            attention_output_v = attention_output_v.permute(0, 1, 4, 3, 2).contiguous()  # b*f, head, c'/head, h, w
            attention_output_v = attention_output_v.view(-1, self.dim_prime, h, w)  # b*f, c', h, w

            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))  # b*f, c, h, w
        else:
            attention_output_h = self.attn(query_h, key_h, value_h)  # b*f, head, h, w*c'/head
            attention_output_v = self.attn(query_v, key_v, value_v)  # b*f, head, w, h*c'/head

            attention_output_h = attention_output_h.view(-1, self.head_num, h, w, self.head_dim)  # b*f, head, h, w, c'/head
            attention_output_h = attention_output_h.permute(0, 1, 4, 2, 3).contiguous()  # b*f, head, c'/head, h, w
            attention_output_h = attention_output_h.view(-1, self.dim_prime, h, w)  # b*f, c', h, w

            attention_output_v = attention_output_v.view(-1, self.head_num, w, h, self.head_dim)  # b*f, head, w, h, c'/head
            attention_output_v = attention_output_v.permute(0, 1, 4, 3, 2).contiguous()  # b*f, head, c'/head, h, w
            attention_output_v = attention_output_v.view(-1, self.dim_prime, h, w)  # b*f, c', h, w
            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=1))  # b*f, c, h, w

        x = attn_out + hx  # b*f, c, h, w

        x = x.view(-1, c, h * w).permute(0, 2, 1).contiguous()  # b*f, h*w, c
        hx = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + hx
        x = x.permute(0, 2, 1).contiguous()  # b*f, c, h*w
        x = x.view(-1, c, h, w)  # b*f, c, h, w
        x = x.view(b, f, c, h, w)

        return x

class Temporal_Inter_SA(nn.Module):
    def __init__(self, dim, head_num):
        super(Temporal_Inter_SA, self).__init__()
        self.dim_prime = dim // 2
        self.head_num = head_num
        self.head_dim = self.dim_prime // self.head_num
        self.attention_norm = nn.LayerNorm(dim)

        self.conv_h = nn.Conv2d(self.dim_prime, 3 * self.dim_prime, kernel_size=1, padding=0)
        self.conv_v = nn.Conv2d(self.dim_prime, 3 * self.dim_prime, kernel_size=1, padding=0)

        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = Mlp(dim)
        self.fuse_out = nn.Conv2d(dim, dim, kernel_size=1, padding=0)
        self.attn = Attention()

    def forward(self, x):
        # x: b, f, c, h, w
        b, f, c, h, w = x.size()
        hx = x  # b, f, c, h, w

        x = x.view(-1, c, h, w)  # b*f, c, h, w
        x = x.view(-1, c, h * w).permute(0, 2, 1).contiguous()  # b*f, h*w, c
        x = self.attention_norm(x).permute(0, 2, 1).contiguous()  # b*f, c, h*w
        x = x.view(-1, c, h, w)   # b*f, c, h, w
        x_input = torch.chunk(x, 2, dim=1)  # 2, b*f, c', h, w

        feature_h = self.conv_h(x_input[0]).view(b, f, 3*self.dim_prime, h, w)  # b, f, 3c', h, w
        feature_h = feature_h.view(b, f, 3, self.dim_prime, h, w)  # b, f, 3, c', h, w (q, k, v)
        feature_h = feature_h.view(b, f, 3, self.head_num, self.head_dim, h, w)  # b, f, 3, head, c'/head, h, w
        feature_h = feature_h.permute(0, 2, 3, 5, 1, 6, 4).contiguous()  # b, 3, head, h, f, w, c'/head
        feature_h = feature_h.view(b, 3, self.head_num, h, f, w * self.head_dim)  # b, 3, head, h, f, w*c'/head
        query_h, key_h, value_h = feature_h[:, 0], feature_h[:, 1], feature_h[:, 2]  # b, head, h, f, w*c'/head

        feature_v = self.conv_v(x_input[1]).view(b, f, 3*self.dim_prime, h, w)  # b, f, 3c', h, w
        feature_v = feature_v.view(b, f, 3, self.dim_prime, h, w)  # b, f, 3, c', h, w (q, k, v)
        feature_v = feature_v.view(b, f, 3, self.head_num, self.head_dim, h, w)  # b, f, 3, head, c'/head, h, w
        feature_v = feature_v.permute(0, 2, 3, 6, 1, 5, 4).contiguous()  # b, 3, head, w, f, h, c'/head
        feature_v = feature_v.view(b, 3, self.head_num, w, f, h * self.head_dim)  # b, 3, head, w, f, h*c'/head
        query_v, key_v, value_v = feature_v[:, 0], feature_v[:, 1], feature_v[:, 2]  # b, head, w, f, h*c'/head

        if h == w:
            query = torch.cat((query_h, query_v), dim=0)  # 2*b, head, (h or w), f, D
            key = torch.cat((key_h, key_v), dim=0)  # 2*b, head, (h or w), f, D
            value = torch.cat((value_h, value_v), dim=0)  # 2*b, head, (h or w), f, D
            attention_output = self.attn(query, key, value)  # 2*b, head, (h or w), f, D
            attention_output = torch.chunk(attention_output, 2, dim=0)  # 2, b, head, (h or w), f, D
            attention_output_h = attention_output[0]  # b, head, h, f, w*c'/head
            attention_output_v = attention_output[1]  # b, head, w, f, h*c'/head

            attention_output_h = attention_output_h.view(b, self.head_num, h, f, w, self.head_dim)  # b, head, h, f, w, c'/head
            attention_output_h = attention_output_h.permute(0, 3, 1, 5, 2, 4).contiguous()  # b, f, head, c'/head, h, w
            attention_output_h = attention_output_h.view(b, f, self.dim_prime, h, w)  # b, f, c', h, w

            attention_output_v = attention_output_v.view(b, self.head_num, w, f, h, self.head_dim)  # b, head, w, f, h, c'/head
            attention_output_v = attention_output_v.permute(0, 3, 1, 5, 4, 2).contiguous()  # b, f, head, c'/head, h, w
            attention_output_v = attention_output_v.view(b, f, self.dim_prime, h, w)  # b, f, c', h, w

            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=2).view(-1, c, h, w))  # b*f, c, h, w
        else:
            attention_output_h = self.attn(query_h, key_h, value_h)  # b, head, h, f, w*c'/head
            attention_output_v = self.attn(query_v, key_v, value_v)  # b, head, w, f, h*c'/head

            attention_output_h = attention_output_h.view(b, self.head_num, h, f, w, self.head_dim)  # b, head, h, f, w, c'/head
            attention_output_h = attention_output_h.permute(0, 3, 1, 5, 2, 4).contiguous()  # b, f, head, c'/head, h, w
            attention_output_h = attention_output_h.view(b, f, self.dim_prime, h, w)  # b, f, c', h, w

            attention_output_v = attention_output_v.view(b, self.head_num, w, f, h, self.head_dim)  # b, head, w, f, h, c'/head
            attention_output_v = attention_output_v.permute(0, 3, 1, 5, 4, 2).contiguous()  # b, f, head, c'/head, h, w
            attention_output_v = attention_output_v.view(b, f, self.dim_prime, h, w)  # b, f, c', h, w

            attn_out = self.fuse_out(torch.cat((attention_output_h, attention_output_v), dim=2).view(-1, c, h, w))  # b*f, c, h, w

        x = attn_out + hx.view(-1, c, h, w)  # b*f, c, h, w
        x = x.view(-1, c, h * w).permute(0, 2, 1).contiguous()  # b*f, h*w, c
        hx = x

        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + hx
        x = x.permute(0, 2, 1).contiguous()  # b*f, c, h*w
        x = x.view(-1, c, h, w)  # b*f, c, h, w
        x = x.view(b, f, c, h, w)  # b*f, c, h, w

        return x


class Spatial_Temporal_Stripformer(nn.Module):
    def __init__(self, dim, head_num):
        super(Spatial_Temporal_Stripformer, self).__init__()
        self.spatial_stripformer = Spatial_Inter_SA(dim, head_num)
        self.temporal_stripformer = Temporal_Inter_SA(dim, head_num)
    def forward(self, x):
        hx = self.spatial_stripformer(x)
        hx = self.temporal_stripformer(hx)
        return hx


class Video_Stripformer(nn.Module):
    def __init__(self, dim_1=64, dim_2=128, dim_3=256, head_num=8):
        super(Video_Stripformer, self).__init__()
        self.encoder = Embeddings(dim_1, dim_2, dim_3)
        self.activation = nn.LeakyReLU(0.2, True)

        self.block_1 = Spatial_Temporal_Stripformer(dim_3, head_num)
        self.block_2 = Spatial_Temporal_Stripformer(dim_3, head_num)
        self.block_3 = Spatial_Temporal_Stripformer(dim_3, head_num)
        self.block_4 = Spatial_Temporal_Stripformer(dim_3, head_num)

        self.sideout_conv = nn.Conv2d(dim_3, 3, kernel_size=3, padding=1)
        self.decoder = Embeddings_output(dim_1, dim_2, dim_3)

    def forward(self, x):

        hx = self.encoder(x)

        hx = self.block_1(hx)
        hx = self.block_2(hx)
        hx = self.block_3(hx)
        hx = self.block_4(hx)

        b, f, c, h, w = hx.size()
        sideouts = (self.sideout_conv(hx.view(-1, c, h, w)).view(b, f, 3, h, w)) + x_quarter

        results = self.decoder(hx) + x

        return results, sideouts

class Video_Stripformer(nn.Module):
    def __init__(self, dim_1=64, dim_2=128, dim_3=256, head_num=8):
        super(Video_Stripformer, self).__init__()
        self.encoder = Embeddings(dim_1, dim_2, dim_3)
        self.activation = nn.LeakyReLU(0.2, True)

        self.block_1 = Spatial_Temporal_Stripformer(dim_3, head_num)
        self.block_2 = Spatial_Temporal_Stripformer(dim_3, head_num)
        self.block_3 = Spatial_Temporal_Stripformer(dim_3, head_num)
        self.block_4 = Spatial_Temporal_Stripformer(dim_3, head_num)

        self.decoder = Embeddings_output(dim_1, dim_2, dim_3)

    def forward(self, x, All=True):

        hx = self.encoder(x)

        hx = self.block_1(hx)
        hx = self.block_2(hx)
        hx = self.block_3(hx)
        hx = self.block_4(hx)

        b, f, c, h, w = hx.size()

        if All:
            results = self.decoder(hx.contiguous()) + x
        else:
            results = self.decoder(hx[:, 1:f - 1].contiguous()) + x[:, 1:f - 1].contiguous()

        return results






