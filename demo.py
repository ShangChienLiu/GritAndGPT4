'''
@kinanmartin 2023/08/02
This code calls GRiT to get object detection and descriptions,
formats the data nicely in text, then passes it in to an LLM
to get a suggestion.
Adapted from original GRiT demo.py
'''

import argparse
import multiprocessing as mp
import os
import time
import cv2
import tqdm
import sys

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger

sys.path.insert(0, 'third_party/CenterNet2/projects/CenterNet2/')
from centernet.config import add_centernet_config
from grit.config import add_grit_config

from grit.predictor import VisualizationDemo

import openai

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import backoff

def set_openai_key():
    openai.api_type = ""
    openai.api_version = ""
    openai.api_base ="https://nlpgroupapi1.openai.azure.com/" # "https://agentgpt.openai.azure.com/" #
    openai.api_key = ""
set_openai_key()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(10))
def call_chatgpt(chatgpt_messages, max_tokens=40, model="gpt-35-turbo"):
    response = openai.ChatCompletion.create(engine=model, messages=chatgpt_messages,#[chatgpt_messages],
    temperature=0.7,
    max_tokens=max_tokens,
    top_p=0.95,
    frequency_penalty=1.2,
    presence_penalty=0,
    stop=None)
    reply = response['choices'][0]['message']['content']
    total_tokens = response['usage']['total_tokens']
    return reply, total_tokens

def prepare_chatgpt_message(pretty_predictions, driver_question=None, few_shot=False):

    if not few_shot:
        chatgpt_messages = [
            {"role": "system", "content": "You are a helpful and intelligent driving assistant. Given a list of object coordinates and their descriptions visible from the front camera of the car, your task is to provide helpful driving suggestions, tips, advice, or warnings to the driver."},
            # {"role": "system", "content": "You are a helpful and intelligent driving assistant. Given a list of object coordinates and their descriptions visible from the front camera of the car, your task is to provide an engaging and informative answer to the driver's questions based on what you see."},
            {"role": "user", "content": pretty_predictions},
            # {"role": "user", "content": "You are driving through the city of Kaohsiung, Taiwan. You are located at: Port of Kaoshiung Entrance. The driver has asked the question: 'What is this big gate with signs'?" + '\n' + pretty_predictions},
        ]
    else:
        chatgpt_messages = [
            {"role": "system", "content": "You are a helpful and intelligent driving assistant. Given a list of object coordinates and their descriptions visible from the front camera of the car, your task is to provide helpful driving suggestions, tips, advice, or warnings to the driver."},
            {"role": "user", "content": "'Image size: 1920x1080 \\nList of object coordinates [x1, y1, x2, y2] and object descriptions:\\n[307, 112, 951, 531] silver car parked on the street\\n[60, 131, 243, 293] the car is gray in color\\n[176, -168, 303, -45] a green and white sign\\n[-415, 263, -146, 535] red line on the road\\n[-959, 225, -196, 533] a row of bushes\\n[267, 144, 514, 332] white car parked on street\\n[-177, 163, -114, 257] motorcycle parked on the sidewalk\\n[-740, -536, 86, 279] green tree on the sidewalk\\n[-939, -538, -739, 334] a tall wooden pole\\n[-252, 89, -205, 145] red and white stop sign\\n[672, 134, 959, 267] the windshield of a car\\n[715, 320, 957, 455] red tail light on a car\\n'"},
            {"role": "assistant", "content": "Be aware of the cars parked to your right and the motorcycle up ahead!"},
            {"role": "user", "content": pretty_predictions},
        ]

    return chatgpt_messages

# constants
WINDOW_NAME = "GRiT"

def setup_cfg(args):
    cfg = get_cfg()
    if args.cpu:
        cfg.MODEL.DEVICE="cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    if args.test_task:
        cfg.MODEL.TEST_TASK = args.test_task
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--cpu", action='store_true', help="Use CPU only.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--output_preds",
        help="A file or directory to save output predictions (text). ",
    )
    parser.add_argument(
        "--LLM",
        help="Name of LLM to call for suggestions."
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--test-task",
        type=str,
        default='',
        help="Choose a task to have GRiT perform",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def prettify_predictions(predictions, center_origin=True):
    coordinates = predictions['instances'].get_fields()['pred_boxes'].tensor # Nx4, (x1, y1, x2, y2)
    pred_object_descriptions = predictions['instances'].get_fields()['pred_object_descriptions'].data

    H, W = predictions['instances']._image_size

    if center_origin:
        coordinates[:, 0::2] -= W / 2
        coordinates[:, 1::2] -= H / 2

    coordinates = coordinates.int().tolist()
    out = f'Image size: {W}x{H} \n'
    out += 'List of object coordinates [x1, y1, x2, y2] and object descriptions:\n'
    for coord, desc in zip(coordinates, pred_object_descriptions):
        out += f'{str(coord)} {desc}\n'

    print(out)
    return out


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)

    if args.input:
        for path in tqdm.tqdm(os.listdir(args.input[0]), disable=not args.output):
            img = read_image(os.path.join(args.input[0], path), format="BGR")
            start_time = time.time()
            predictions, visualized_output = demo.run_on_image(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            if args.output_preds:
                if not os.path.exists(args.output_preds):
                    os.mkdir(args.output_preds)
                if os.path.isdir(args.output_preds):
                    assert os.path.isdir(args.output_preds), args.output_preds
                    out_preds_filename = os.path.join(args.output_preds, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output_preds"
                    out_preds_filename = args.output_preds

                pretty_preds = prettify_predictions(predictions)
                # breakpoint()
                with open(f'{os.path.splitext(out_preds_filename)[0]}.txt', "w") as fp:
                    fp.write(pretty_preds)

                if args.LLM:
                    chatgpt_messages = prepare_chatgpt_message(pretty_preds, driver_question=None, few_shot=True)
                    reply, _ = call_chatgpt(chatgpt_messages, max_tokens=100, model=args.LLM)
                    print(reply)
                    with open(f'{os.path.splitext(out_preds_filename)[0]}_{args.LLM}_out.txt', "w") as fp:
                        fp.write(reply)

            if args.output:
                if not os.path.exists(args.output):
                    os.mkdir(args.output)
                if os.path.isdir(args.output):
                    assert os.path.isdir(args.output), args.output
                    out_filename = os.path.join(args.output, os.path.basename(path))
                else:
                    assert len(args.input) == 1, "Please specify a directory with args.output"
                    out_filename = args.output
                visualized_output.save(out_filename)
            else:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
                if cv2.waitKey(0) == 27:
                    break  # esc to quit
