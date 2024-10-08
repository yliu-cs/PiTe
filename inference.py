import os
import torch
from torch import nn
from pite import PiTeLlamaForCausalLM
from argparse import ArgumentParser, Namespace
from utils import conversation as llava_conversation
from utils.mm_utils import tokenizer_video_token, KeywordsStoppingCriteria
from utils.misc import ignore_warnings, load_lora, disable_torch_init, extract_video
from transformers import AutoTokenizer, PreTrainedTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection


def get_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--fps", type=int, default=100)
    parser.add_argument("--model_scale", type=str, default="7b")
    parser.add_argument("--model_base", type=str, default=os.path.join(os.getcwd(), "checkpoint", "vicuna-7b-v1.5"))
    parser.add_argument("--stage1", type=str, default=os.path.join(os.getcwd(), "checkpoint", "pite-vicuna-7b-v1.5-stage1"))
    parser.add_argument("--stage2", type=str, default=os.path.join(os.getcwd(), "checkpoint", "pite-vicuna-7b-v1.5-stage2"))
    parser.add_argument("--stage3", type=str, default=os.path.join(os.getcwd(), "checkpoint", "pite-vicuna-7b-v1.5-stage3"))
    parser.add_argument("--clip_path", type=str, default=os.path.join(os.getcwd(), "checkpoint", "clip-vit-large-patch14"))
    parser.add_argument("--pretrain_vision_adapter", type=str, default=None)
    parser.add_argument("--query", type=str, default=None, required=True)
    args = parser.parse_args()
    rev_scale = "7b" if args.model_scale == "13b" else "13b"
    if args.model_scale not in args.model_base:
        args.model_base = args.model_base.replace(rev_scale, args.model_scale)
    if args.model_scale not in args.stage1:
        args.stage1 = args.stage1.replace(rev_scale, args.model_scale)
    if args.model_scale not in args.stage2:
        args.stage2 = args.stage2.replace(rev_scale, args.model_scale)
    if args.model_scale not in args.stage3:
        args.stage3 = args.stage3.replace(rev_scale, args.model_scale)
    args.pretrain_vision_adapter = os.path.join(args.stage2, "vision_adapter.bin")
    if not os.path.exists(args.pretrain_vision_adapter):
        args.pretrain_vision_adapter = os.path.join(args.stage1, "vision_adapter.bin")
    return args


def load_pretrained_model(
    args: Namespace
    , stage1: str = None
    , stage2: str = None
    , stage3: str = None
) -> tuple[PreTrainedTokenizer, nn.Module]:
    kwargs = {"torch_dtype": torch.float16}
    model_base = args.model_base
    print("\033[33mLoading PiTe from base model...\033[0m")
    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    model = PiTeLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
    token_num, token_dim = model.lm_head.out_features, model.lm_head.in_features
    if model.lm_head.weight.shape[0] != token_num:
        model.lm_head.weight = nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))
        model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, token_dim, device=model.device, dtype=model.dtype))
    model.get_model().initialize_mlp_modules(args)
    if stage1 is not None:
        print("\033[33mLoading stage1 weights...\033[0m")
        model = load_lora(model, stage1)
        print("\033[33mMerging stage1 weights...\033[0m")
        model = model.merge_and_unload()
        if stage2 is not None:
            print("\033[33mLoading stage2 weights...\033[0m")
            model = load_lora(model, stage2)
            print("\033[33mMerging stage2 weights...\033[0m")
            model = model.merge_and_unload()
            if stage3 is not None:
                print("\033[33mLoading stage3 weights...\033[0m")
                model = load_lora(model, stage3)
                print("\033[33mMerging stage3 weights...\033[0m")
                model = model.merge_and_unload()
    return tokenizer, model


def inference(
    model: nn.Module
    , videos: torch.Tensor
    , query: str
    , tokenizer: PreTrainedTokenizer
) -> str:
    conv = llava_conversation.conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_video_token(
        prompt=prompt
        , tokenizer=tokenizer
        , return_tensors="pt"
    ).unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != llava_conversation.SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids
            , videos=videos.unsqueeze(0).cuda()
            , do_sample=True
            , temperature=0.05
            , num_beams=1
            , max_new_tokens=1024
            , stopping_criteria=[stopping_criteria]
        )
    input_token_len = input_ids.size(1)
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f"\033[33m[Warning] {n_diff_input_output} output_ids are not the same as the input_ids\033[0m")
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    return outputs


def main(args: Namespace) -> None:
    disable_torch_init()
    tokenizer, model = load_pretrained_model(args, args.stage1, args.stage2, args.stage3)
    model = model.cuda()
    model.to(torch.float16)
    clip_model = CLIPVisionModelWithProjection.from_pretrained(args.clip_path)
    clip_model = clip_model.cuda()
    processor = CLIPImageProcessor.from_pretrained(args.clip_path)
    try:
        images = extract_video(args.video_path, fps=args.fps)
    except:
        print(f"\033[31mFailed to load {args.video_path}...\033[0m")
    inputs = processor(images=images, return_tensors="pt").to(clip_model.device)
    with torch.no_grad():
        outputs = clip_model(**inputs)
    features = outputs.image_embeds
    features = features.to(torch.float16)
    query = "describe the video" if args.query is None else args.query
    answer = inference(model, features, f"<video>\n{query}", tokenizer)
    print(answer)


if __name__ == "__main__":
    ignore_warnings()
    args = get_args()
    main(args=args)