import pytest


def test_policy_forward_candidate_shapes():
    torch = pytest.importorskip("torch")

    from gnn_policies import ActorCriticPolicy

    policy = ActorCriticPolicy(
        feature_dim=4,
        num_global_ops=2,
        num_local_sizes=3,
        num_violation_keys=1,
    )
    policy.eval()

    x = torch.randn(5, 4)
    edge_index = torch.zeros((2, 0), dtype=torch.long)

    global_feats = torch.zeros((0, 11))
    local_feats = torch.zeros((0, 10))
    graph_h, global_logits, local_logits, size_logits = policy(
        x, edge_index, global_cand_feats=global_feats, local_cand_feats=local_feats
    )
    assert graph_h.ndim == 1
    assert global_logits.shape == (0,)
    assert local_logits.shape == (0,)
    assert size_logits.shape == (3,)

    x2 = torch.randn(2, 4)
    edge_index2 = torch.zeros((2, 1), dtype=torch.long)
    global_feats2 = torch.randn(3, 11)
    local_feats2 = torch.randn(2, 10)
    _, global_logits2, local_logits2, size_logits2 = policy(
        x2, edge_index2, global_cand_feats=global_feats2, local_cand_feats=local_feats2
    )
    assert global_logits2.shape == (3,)
    assert local_logits2.shape == (2,)
    assert size_logits2.shape == (3,)