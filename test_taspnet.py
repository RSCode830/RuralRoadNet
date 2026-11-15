import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from scanning import affine_scan, stripe_groups, TopologyMamba
from taspnet import TaSPNet
from priors import structure_score, fuse_proposals
from losses import objective, cldice
from metrics import from_counts, Metrics
from data import Roads, check_splits

torch.set_num_threads(2)


def test_associative_scan_values_and_gradients():
    torch.manual_seed(1)
    a = torch.rand(2, 17, 3, 4, dtype=torch.double, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    reference, fast = affine_scan(a, b, True), affine_scan(a, b)
    assert torch.allclose(reference, fast, atol=1e-12)
    gr = torch.autograd.grad(reference.square().sum(), (a, b), retain_graph=True)
    gf = torch.autograd.grad(fast.square().sum(), (a, b))
    for first, second in zip(gr, gf):
        assert torch.allclose(first, second, atol=1e-10)


@pytest.mark.parametrize('shift', [0, 2])
def test_stripe_partition_and_inverse(shift):
    labels = np.random.RandomState(2).randint(0, 4, (13, 17))
    angles = [-np.pi/2, -np.pi/4, 0, np.pi/4]
    groups = stripe_groups(labels, angles, 4, shift)
    ids = np.concatenate([indices for _, indices in groups])
    assert np.array_equal(np.sort(ids), np.arange(labels.size))
    values = np.arange(labels.size)*3
    restored = np.zeros_like(values)
    for direction, indices in groups:
        assert (labels.ravel()[indices] == direction).all()
        restored[indices] = values[indices]
    assert np.array_equal(restored, values)


@pytest.mark.parametrize('shape', [(16, 16), (17, 19)])
def test_network_backward(shape):
    torch.manual_seed(2)
    model = TaSPNet(widths=(4, 8, 12), depths=(2, 2), decoder_depth=1, state_dim=2, stripe_width=4)
    image, prior = torch.rand(2, 3, *shape), torch.rand(2, 1, *shape)
    result = model(image, prior)
    assert result['logits'].shape == (2, 1, *shape)
    loss = objective(result, (torch.rand_like(prior) > .7).float(), skeleton_iterations=3)
    loss.backward()
    for name, value in model.named_parameters():
        assert value.grad is not None, name
        assert torch.isfinite(value.grad).all(), name
    assert model.encoder[0][0].mamba.orientation[-1].weight.grad.abs().sum() > 0
    assert model.decoder[0][0].global_mlp[-1].weight.grad.abs().sum() > 0


def test_batch_isolation_and_padding():
    torch.manual_seed(3)
    block = TopologyMamba(4, state_dim=2, stripe_width=3).eval()
    x = torch.rand(2, 4, 9, 11)
    with torch.no_grad():
        together = block(x)
        separately = torch.cat([block(item[None]) for item in x])
    assert torch.allclose(together, separately, atol=2e-6)


def test_geometry_scores_and_literal_sigmoid():
    thin = np.zeros((32, 32)); thin[15:17, 2:30] = 1
    square = np.zeros_like(thin); square[8:24, 8:24] = 1
    assert structure_score(thin) > structure_score(square)
    prior = fuse_proposals([thin, square], thin.shape, top_k=1)
    assert np.allclose(prior, 1/(1+np.exp(-thin)))
    assert np.all(fuse_proposals([], thin.shape) == .5)
    with pytest.raises(ValueError):
        fuse_proposals([np.full((2, 2), np.nan)], (2, 2))


def test_topology_and_empty_foreground():
    truth = torch.zeros(1, 1, 16, 16); truth[:, :, 8, 1:15] = 1
    broken = truth.clone(); broken[:, :, 8, 6:10] = 0
    assert cldice(truth, truth).item() > cldice(broken, truth).item()
    empty = torch.zeros_like(truth)
    assert cldice(empty, empty).item() == pytest.approx(1)
    assert cldice(torch.ones_like(empty), empty).item() == pytest.approx(0)


def test_metrics_counts():
    m = from_counts(3, 1, 2)
    assert m['precision'] == .75 and m['recall'] == .6
    assert m['f1'] == pytest.approx(2/3) and m['road_iou'] == .5
    metric = Metrics()
    metric.update(torch.full((1, 1, 4, 4), -20.), torch.zeros(1, 1, 4, 4))
    assert metric.compute()['macro_percent']['road_iou'] == 100


def test_data_contract_and_group_leakage(tmp_path):
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(tmp_path/'image.png')
    Image.fromarray(np.zeros((8, 8), np.uint8)).save(tmp_path/'mask.png')
    np.save(str(tmp_path/'prior.npy'), np.ones((16, 16), np.float32)*.5)
    rows = [dict(image='image.png', mask='mask.png', prior='prior.npy', prior_kind='sam', group='scene1')]
    path = tmp_path/'manifest.json'; path.write_text(json.dumps(rows))
    cfg = dict(image_size=16, prior_kind='sam')
    dataset = Roads(path, cfg, True)
    image, mask, prior = dataset[0]
    assert image.shape == (3, 16, 16) and mask.sum() == 0 and prior.mean() == .5
    with pytest.raises(ValueError):
        check_splits(dataset, dataset)
    with pytest.raises(ValueError):
        Roads(path, dict(cfg, prior_kind='synthetic_demo'))
    Image.fromarray(np.full((8, 8), 127, np.uint8)).save(tmp_path/'mask.png')
    with pytest.raises(ValueError):
        dataset[0]


def test_prior_changes_prediction_and_checkpoint(tmp_path):
    torch.manual_seed(5)
    cfg = dict(widths=(4, 8), depths=(1,), decoder_depth=1, state_dim=2)
    model = TaSPNet(**cfg).eval()
    image = torch.rand(1, 3, 8, 8)
    with torch.no_grad():
        a = model(image, torch.zeros(1, 1, 8, 8))['logits']
        b = model(image, torch.ones(1, 1, 8, 8))['logits']
    assert (a-b).abs().max() > 1e-6
    path = tmp_path/'weights.pt'; torch.save(model.state_dict(), path)
    restored = TaSPNet(**cfg).eval(); restored.load_state_dict(torch.load(path))
    with torch.no_grad():
        assert torch.equal(a, restored(image, torch.zeros(1, 1, 8, 8))['logits'])
