import numpy as np
import pytest
import torch
import torchvision

from vrt.postprocess import cxcywh_to_xyxy, nms_reference, postprocess


def random_boxes(n, rng):
    xy = rng.uniform(0, 600, size=(n, 2))
    wh = rng.uniform(5, 150, size=(n, 2))
    return np.concatenate([xy, xy + wh], axis=1).astype(np.float32)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("iou", [0.3, 0.45, 0.65])
def test_reference_nms_matches_torchvision(seed, iou):
    rng = np.random.default_rng(seed)
    boxes = random_boxes(500, rng)
    scores = rng.uniform(size=500).astype(np.float32)
    ours = nms_reference(boxes, scores, iou)
    theirs = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou).numpy()
    np.testing.assert_array_equal(ours, theirs)


def test_reference_nms_suppresses_duplicates():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    np.testing.assert_array_equal(nms_reference(boxes, scores, 0.5), [0, 2])


def test_cxcywh_to_xyxy():
    b = torch.tensor([[10.0, 20.0, 4.0, 6.0]])
    torch.testing.assert_close(cxcywh_to_xyxy(b), torch.tensor([[8.0, 17.0, 12.0, 23.0]]))


def test_postprocess_filters_scales_and_keeps_classes_apart():
    pred = torch.zeros(4, 85)
    # Two heavily overlapping boxes of different classes both survive (class-aware NMS),
    # a same-class duplicate is suppressed, and a low-score box is filtered out.
    pred[0, :5] = torch.tensor([100, 100, 50, 50, 0.9]); pred[0, 5 + 3] = 1.0
    pred[1, :5] = torch.tensor([101, 101, 50, 50, 0.8]); pred[1, 5 + 3] = 1.0
    pred[2, :5] = torch.tensor([100, 100, 50, 50, 0.7]); pred[2, 5 + 7] = 1.0
    pred[3, :5] = torch.tensor([300, 300, 50, 50, 0.1]); pred[3, 5 + 1] = 1.0
    dets = postprocess(pred, scale=0.5, conf_thresh=0.3, iou_thresh=0.45)
    assert dets.shape == (2, 6)
    assert sorted(dets[:, 5].tolist()) == [3.0, 7.0]
    np.testing.assert_allclose(dets[0, :4], [150, 150, 250, 250])  # (75..125) / 0.5
