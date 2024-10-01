import torch
import argparse
from eval.models.qa_model import ImageQAModel, VideoQAModel
from eval.models.qa_model.videoqa_model import videoqa_models
from eval.models.qa_model.imageqa_model import imageqa_models
from eval.eval_utils.task_evaluator import PhysionBenchEvaluator, task_split
from eval.eval_utils.caculate_core import calculate_accuracy, print_accuracies


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='gpt4o', help="Select the model name")
    parser.add_argument("--dataset_path", type=str, default='./eval/physbench',
                        help="data you put USC-GVL/PhysBench")
    parser.add_argument("--split", type=str, default='val', choices=['val', 'test'],
                        help="Choose between 'val' or 'test' split")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    model_name = args.model_name
    dataset_path = args.dataset_path

    if model_name in imageqa_models.keys():
        model = ImageQAModel(model_name=model_name, precision=torch.float16)
    elif model_name in videoqa_models.keys():
        model = VideoQAModel(model_name=model_name)
    else:
        raise ValueError(f"Model '{model_name}' is not supported")

    task_evaluator = PhysionBenchEvaluator(
        model=model.model,
        mode=task_split[model_name],
        dataset_path=dataset_path,
        model_name=model_name,
        resume=True,
        sample_ratio=None,
        split=args.split
    )

    task_evaluator.test()

    if args.split == 'val':
        accuracies = calculate_accuracy(val_annotation_file='./eval/physbench/val_answer.json',
                                        user_submission_file=task_evaluator.result_file)
        print_accuracies(accuracies, name=model_name)
