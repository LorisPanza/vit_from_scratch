import torch
from torch import nn

from einops.layers.torch import Rearrange



class PatchEmbedding(nn.Module):
    """
    Convert the image into patches and then project them into a vector space.
    img size = (batch,channel,height,width)
    """
    def __init__(self, patch_size, in_channels, hidden_size, img_size) :
        super().__init__()
        self.in_channels = in_channels
        assert img_size[-1] % patch_size == 0 and img_size[-2] % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
  
        self.out_channel = ((img_size[-3]*img_size[-2]*img_size[-1])//patch_size**2)
        self.hidden_size = hidden_size

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=self.out_channel, kernel_size=patch_size, stride=patch_size, padding=0)
        self.linear = nn.Linear(in_features=self.out_channel, out_features=self.hidden_size)

    def forward(self,x):
        # (Batch, channels, width, height) -> (Batch, patch_number, patch_hidden_dim)
        x = self.conv(x)
        x = torch.flatten(x,start_dim = 2,end_dim=3)
        x = x.permute(0,2,1)
        x = self.linear(x)
        return x
    
def get_positional_encoding(embedding_size, sequence):
    """
    # timestamp represents the vector time that should be expaned in embedding size dimensions, it should be a tensor vector
    # timestamp -> (sequence, embeddingsize)
    """

    embedding_range_half = torch.arange(0,int(embedding_size/2),step=1)
    denominator = 10000**((2*embedding_range_half)/embedding_size)

    timesteps_dimensionality_half = sequence[:,None].repeat(1, embedding_size//2)/denominator

    cos_seq = torch.cos(timesteps_dimensionality_half)
    sin_seq = torch.sin(timesteps_dimensionality_half)

    positional_embed = torch.cat(tensors=[cos_seq,sin_seq],dim=-1)

    return positional_embed


class Embeddings(nn.Module):
    """
    Combine the patch embeddings with the class token and position embeddings.
    """
    def __init__(self, hidden_size, n_patches):
        super().__init__()
        self.classification_token = nn.Parameter(torch.rand(1,hidden_size))
        self.positional_embedding = nn.Parameter(get_positional_encoding(hidden_size,torch.arange(0,n_patches+1,step=1)))
        self.positional_embedding.requires_grad = False


    def forward(self,x):
        #(Batch, patch_number, patch_hidden_dim) -> (Batch, patch_number+1, patch_hidden_dim)
        batch_size,n,_ = x.shape 
        cls_token = self.classification_token.expand(batch_size,1,-1)
        x = torch.cat((cls_token,x),dim=1)
        x = x + self.positional_embedding
        return x


class Encoder(nn.Module):
    #(Batch, Patch_number+1, Patch_hidden_dim)

    def __init__(self,hidden_size,n_heads):
        super().__init__()
        self.hidden_size = hidden_size

        self.layer_normalization1 = nn.LayerNorm(self.hidden_size)
        self.layer_normalization2 = nn.LayerNorm(self.hidden_size)
        self.msa = Attention(n_heads,self.hidden_size)

        self.mlp =nn.Sequential(
         nn.Linear(self.hidden_size,self.hidden_size),
         nn.GELU(),
         nn.Linear(self.hidden_size,self.hidden_size))

    def forward(self,x):
        #(Batch, Patch_number+1, Patch_hidden_dim) -> (Batch, Patch_number+1, Patch_hidden_dim)
        out = x + self.msa(self.layer_normalization1(x))
        out = self.layer_normalization2(out)
        out = out + self.mlp(out)
        return out


class Attention(nn.Module):
    def __init__(self,n_heads,hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_heads = n_heads

        if((3*self.hidden_size)%self.n_heads!=0):
            print("Numero di head incompatibile")

        self.qkv = nn.Linear(hidden_size, 3*self.hidden_size*self.n_heads)
        self.softmax = nn.Softmax(dim=-1)
        self.linear = nn.Linear(self.n_heads * self.hidden_size, self.hidden_size)


    def forward(self,x):
        batch_size = x.shape[0]
        patch_n = x.shape[1]

        #(Batch, Patch_number+1, Patch_hidden_dim) -> #(B, P+1, H*3)
        qkv = self.qkv(x)

        qkv = qkv.view(batch_size, patch_n, self.n_heads, -1)

        qkv = qkv.transpose(1,2)

        q,k,v = torch.chunk(qkv,3,dim=-1)
        
        qk = torch.matmul(q, k.transpose(-1,-2))

        qk_softm = self.softmax(qk/(q.shape[-1]**0.5))

        #(Batch, N_HEADS, P+1,  P+1) -> #(B, N_HEADS, P+1, H)
        attention = torch.matmul(qk_softm,v)

        attention = attention.reshape(batch_size,patch_n, -1)

        attention = self.linear(attention)

        return attention



class Vit(nn.Module):
    def __init__(self, in_channels, hidden_size, img_size,num_classes,patch_size):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_dim = in_channels*patch_size*patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.LayerNorm(self.patch_dim),
            nn.Linear(self.patch_dim, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        #self.patch_embedding = PatchEmbedding(in_channels=self.in_channels, hidden_size=self.hidden_size, img_size=self.img_size,patch_size=self.patch_size)

        self.npatches = (img_size[-1]//patch_size)**2

        self.embedding_pos_cls = Embeddings(hidden_size=self.hidden_size, n_patches=self.npatches)

        self.encoder = Encoder(hidden_size = self.hidden_size, n_heads = 2)


        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_size, num_classes)
        )

    def forward(self,x):
        x = self.to_patch_embedding(x)
        x = self.embedding_pos_cls(x)
        x = self.encoder(x)
        x=x[:,0]
        x=self.mlp(x)
        return x






