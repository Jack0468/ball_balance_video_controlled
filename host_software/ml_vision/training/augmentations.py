from typing import List, Tuple

import albumentations as A

INPUT_SIZE = (128, 128)  # (H, W)


def photometric_ops() -> List[A.BasicTransform]:
    return [
        A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.7),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=15, p=0.5),
    ]


def shadow_ops() -> List[A.BasicTransform]:
    # Motivated by docs/PROJECT_LOGBOOK.md (2026-07-13): shadows were found to be a
    # real confound for grey/black marker detection under naive HSV thresholding.
    return [
        A.RandomShadow(
            shadow_roi=(0, 0, 1, 1),
            num_shadows_limit=(1, 3),
            shadow_dimension=5,
            shadow_intensity_range=(0.4, 0.7),
            p=0.7,
        ),
    ] + photometric_ops()


def geometric_jitter_ops() -> List[A.BasicTransform]:
    # Tests robustness to platform/camera "state" variance (slight translation, scale,
    # rotation). NEVER add A.HorizontalFlip / A.VerticalFlip here -- flipping mirrors
    # the physical board and swaps left/right marker semantics, permanently forbidden
    # per docs/PROJECT_LOGBOOK.md (2026-07-16, "The Only Forbidden Augmentation").
    #
    # train_resnet_expert_tracker.py / train_resnet_2d_tracker.py deliberately removed
    # RandomAffine/RandomPerspective (their comment: augmenting the image while leaving
    # touch_x/touch_y labels untouched forces the model to re-estimate camera homography
    # and degraded tracking accuracy). That failure mode doesn't apply here: callers
    # (SharedVisionDataset / TrialAugmentedDataset) transform the image, mask, AND ball
    # keypoint jointly, so the label always stays correct for whatever the image now shows.
    return [
        A.Affine(translate_percent=(-0.05, 0.05), scale=(0.9, 1.1), rotate=(-10, 10), p=0.8),
        A.Perspective(scale=(0.02, 0.05), p=0.4),
    ]


def blur_ops() -> List[A.BasicTransform]:
    # Matches the GaussianBlur(kernel_size=(5,9), sigma=(0.1,5.0)) used across
    # train_cnn_2d_tracker.py, train_resnet_2d_tracker.py, train_cnn_expert_tracker.py,
    # train_resnet_expert_tracker.py, and temporal_ball_dataset.py -- simulates
    # camera focus/motion blur, the most consistently-used augmentation in this repo.
    return [A.GaussianBlur(blur_limit=(5, 9), sigma_limit=(0.1, 5.0), p=0.4)]


def occlusion_ops() -> List[A.BasicTransform]:
    # Albumentations equivalent of transforms.RandomErasing(p=0.4, scale=(0.02,0.1)),
    # used in train_cnn_2d_tracker.py / train_resnet_2d_tracker.py / temporal_ball_dataset.py.
    # Only blanks pixels in the image -- mask/keypoint targets are left untouched (verified:
    # CoarseDropout does not modify the mask target when fill_mask is left at its default),
    # so this trains the model to infer position/markers from partial visual evidence
    # rather than changing what the "correct" answer is.
    return [
        A.CoarseDropout(
            num_holes_range=(1, 3),
            hole_height_range=(0.02, 0.1),
            hole_width_range=(0.02, 0.1),
            fill=0,
            p=0.4,
        )
    ]


def build_train_transform(input_size: Tuple[int, int] = INPUT_SIZE) -> A.Compose:
    """Production training augmentation: OneOf{photometric, shadow, geometric_jitter}.

    Chosen from host_software/ml_vision/experiments/trial_augmentation_strategies.py's
    8-variant trial (see experiments/results/ANALYSIS_2026-08-11.md) -- this composition
    ranked 2nd (31.5px) at full scale and 3rd (42.2+/-2.5px, lowest variance of any
    augmented variant) at reduced scale. blur/occlusion are deliberately excluded (they
    underperformed baseline in the 5-seed run), and the 3 groups are OneOf'd rather than
    stacked -- stacking everything (the trial's "combined" variant) underperformed the
    individual pieces (38.0px vs 27.8-34.8px), so only one group is ever applied per image.
    """
    ops = photometric_ops() + shadow_ops() + geometric_jitter_ops()
    return A.Compose(
        [A.Resize(*input_size), A.OneOf(ops, p=0.9)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def build_eval_transform(input_size: Tuple[int, int] = INPUT_SIZE) -> A.Compose:
    """Resize-only, no augmentation -- for validation/test splits, so metrics reflect
    generalisation to clean data rather than re-augmented eval noise."""
    return A.Compose(
        [A.Resize(*input_size)],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
