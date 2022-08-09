import torch
import torch.nn as nn

class NCF(nn.Module):
    def __init__(self, args):
        super(NCF, self).__init__()
        self.embed_size = args.embed_size

        # user embed layer
        self.mlp_user = nn.Embedding(args.user_len + 1, self.embed_size)
        self.mf_user = nn.Embedding(args.user_len + 1, self.embed_size)

        # item embed layer
        self.mlp_item = nn.Embedding(args.item_len + 1, self.embed_size)
        self.mf_item = nn.Embedding(args.item_len + 1, self.embed_size)

        # MLP layer
        self.mlp_1 = nn.Linear(self.embed_size * 2, args.hidden_size_1)
        self.mlp_2 = nn.Linear(args.hidden_size_1, args.hidden_size_2)
        self.mlp_3 = nn.Linear(args.hidden_size_2, args.hidden_size_3)

        # prediction
        self.pred = nn.Linear(self.embed_size + args.hidden_size_3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        mlp_user_vec = self.mlp_user(user)
        mf_user_vec = self.mf_user(user)

        mlp_item_vec = self.mlp_item(item)

        mf_item_vec = self.mf_item(item)

        # mlp_layer
        a = torch.cat((mlp_user_vec, mlp_item_vec), 1)
        a = self.mlp_1(a)
        a = self.mlp_2(a)
        a = self.mlp_3(a)

        # gmf_layer
        b = torch.mul(mf_user_vec, mf_item_vec)

        # concat
        x = torch.cat((a, b), 1)
        x = self.pred(x)

        return self.sigmoid(x)

