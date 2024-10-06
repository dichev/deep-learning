from lib.layers import *
import tempfile
from lib import optimizers, layers
from models.transformer_networks import LLaMA2, GPT3


def test_linear_load_state_dict():
    x = torch.randn(10, 16)
    model = layers.Linear(16, 8)
    validate_checkpoint_consistency(model, x)


def test_bn_load_state_dict():
    x = torch.randn(20, 16, 8, 8)
    bn = BatchNorm2d(16)  # has buffers
    validate_checkpoint_consistency(bn, x)


def test_conv_load_state_dict():
    x = torch.randn(20, 16, 8, 8).to('cuda')
    conv = Conv2d(16, 8, 2).to('cuda')
    validate_checkpoint_consistency(conv, x)


def test_LLAMA_load_state_dict():
    model = LLaMA2(vocab_size=128, context_size=16, embed_size=64, hidden_size=64 * 4, n_layers=4, attn_heads=16, attn_kv_groups=0).to('cuda')
    x = torch.randint(128, (10, 16)).to('cuda')
    validate_checkpoint_consistency(model, x)


def test_GPT3_load_state_dict():
    modelA = GPT3(vocab_size=128, context_size=16, embed_size=64, hidden_size=64 * 4, n_layers=4, attn_heads=16, local_attn_block_size=4, dropout=0).to('cuda')
    x = torch.randint(128, (10, 16)).to('cuda')
    validate_checkpoint_consistency(modelA, x)



def validate_checkpoint_consistency(model, x):
    optim = optimizers.AdamW(model.parameters(), lr=.1, weight_decay=0.1, weight_decay_filter=f'weight')

    def train_loop(steps):
        losses = []
        for i in range(steps):
            y = model.forward(x).sum()
            y.backward()
            optim.step().zero_grad()
            losses.append(y.item())
        return torch.tensor(losses)

    # do some pre-training
    train_loop(steps=5)

    # save a checkpoint
    with tempfile.TemporaryFile() as tmp_file:
        torch.save({
            'model': model.state_dict(),
            'optim': optim.state_dict()
        }, tmp_file)
        tmp_file.seek(0)
        checkpoint = torch.load(tmp_file, weights_only=False)  # cloned

    # do some training
    losses_A = train_loop(steps=10)

    # restore the checkpoint and do the same training
    model.load_state_dict(checkpoint['model'])
    optim.load_state_dict(checkpoint['optim'])
    losses_B = train_loop(steps=10)

    # validate the losses are the same
    assert torch.allclose(losses_A, losses_B)

