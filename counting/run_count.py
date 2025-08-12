import torch
import cv2

from boxmot import TRACKERS
from boxmot.tracker_zoo import create_tracker
from boxmot.utils import ROOT
from boxmot.utils.checks import TestRequirements


# __tr = TestRequirements()
# local_path = 'ultralytics'
# __tr.check_packages((local_path,))

from counting.count import counter_YOLO

from ultralytics.utils import LOGGER, ops, colorstr

from functools import partial
from pathlib import Path
import csv
import time
import os
from datetime import datetime


def on_predict_start(predictor, persist=False):
    """
    Initialize trackers for object tracking during prediction.

    Args:
        predictor (object): The predictor object to initialize trackers for.
        persist (bool, optional): Whether to persist the trackers if they already exist. Defaults to False.
    """

    assert (
        predictor.custom_args.tracking_method in TRACKERS
    ), f"'{predictor.custom_args.tracking_method}' is not supported. Supported ones are {TRACKERS}"

    tracking_config = (
        ROOT / "boxmot" / "configs" / (predictor.custom_args.tracking_method + ".yaml")
    )
    trackers = []
    for i in range(predictor.dataset.bs):
        tracker = create_tracker(
            predictor.custom_args.tracking_method,
            tracking_config,
            predictor.custom_args.reid_model,
            predictor.device,
            predictor.custom_args.half,
            predictor.custom_args.per_class,
        )
        # motion only modeles do not have
        if hasattr(tracker, "model"):
            tracker.model.warmup()
        trackers.append(tracker)

    predictor.trackers = trackers


@torch.no_grad()
def run(args):
    counter_yolo = counter_YOLO(args)

    # counter_yolo.predict(source=args.source, stream=False)
    counter_yolo.predictor = counter_yolo._get_predictor(overrides=vars(args))

    counter_yolo.add_callback(
        "on_predict_start", partial(on_predict_start, persist=True)
    )
    counter_yolo.predictor.custom_args = args

    # Setup model
    model = None
    if not counter_yolo.predictor.model:
        counter_yolo.predictor.setup_model(model)

    # Setup source every time predict is called
    source = args.source
    counter_yolo.predictor.setup_source(
        source if source is not None else counter_yolo.predictor.args.source
    )

    # Warmup model
    if not counter_yolo.predictor.done_warmup:
        counter_yolo.predictor.model.warmup(
            imgsz=(
                (
                    1
                    if counter_yolo.predictor.model.pt
                    or counter_yolo.predictor.model.triton
                    else counter_yolo.predictor.dataset.bs
                ),
                3,
                *counter_yolo.predictor.imgsz,
            )
        )
        counter_yolo.predictor.done_warmup = True

    (
        counter_yolo.predictor.seen,
        counter_yolo.predictor.windows,
        counter_yolo.predictor.batch,
        profilers,
    ) = (
        0,
        [],
        None,
        (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile()),
    )
    counter_yolo.predictor.run_callbacks("on_predict_start")

    for batch in counter_yolo.predictor.dataset:
        counter_yolo.predictor.run_callbacks("on_predict_batch_start")
        counter_yolo.predictor.batch = batch
        path, im0s, vid_cap, s = batch

        counter_yolo.frame_number = int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES))

        im0s, im, profilers = counter_yolo.run_pipeline(im0s, path, profilers)

        n = len(im0s)

        for i in range(n):

            counter_yolo.predictor.seen += 1

            # Counting Phase
            with profilers[4]:
                counter_yolo.run_counting(i)

            result = counter_yolo.predictor.results[i]

            result.speed = {
                "preprocess": profilers[0].dt * 1e3 / n,
                "inference": profilers[1].dt * 1e3 / n,
                "postprocess": profilers[2].dt * 1e3 / n,
                "tracking": profilers[3].dt * 1e3 / n,
                "counting": profilers[4].dt * 1e3 / n,
            }

        counter_yolo.predictor.run_callbacks("on_predict_batch_end")

    return counter_yolo, profilers
