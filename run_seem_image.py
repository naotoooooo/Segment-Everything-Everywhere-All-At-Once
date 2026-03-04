# # # --------------------------------------------------------
# # # Minimal SEEM Image Inference Script (No Gradio)
# # # --------------------------------------------------------

# import os
# import torch
# import argparse
# import whisper
# import time
# from PIL import Image

# from modeling.BaseModel import BaseModel
# from modeling import build_model
# from utils.distributed import init_distributed
# from utils.arguments import load_opt_from_config_files
# from utils.constants import COCO_PANOPTIC_CLASSES

# from demo.seem.tasks import interactive_infer_image

# import glob  # 追加

# # ----------------------------
# # Argument
# # ----------------------------
# def parse_option():
#     parser = argparse.ArgumentParser("SEEM Image Inference")
#     parser.add_argument(
#         "--conf_files",
#         default="configs/seem/focall_unicl_lang_demo.yaml",
#         help="path to config file",
#     )
#     parser.add_argument("--input", required=True, help="input image path")
#     parser.add_argument("--text", default=None, help="text prompt")
#     parser.add_argument("--output", default="output/output.png", help="output image path")
#     return parser.parse_args()

# # ----------------------------
# # Main
# # ----------------------------
# def main():
#     args = parse_option()

#     # load config
#     opt = load_opt_from_config_files([args.conf_files])
#     opt = init_distributed(opt)

#     # select pretrained model
#     if "focalt" in args.conf_files:
#         pretrained_pth = "seem_focalt_v0.pt"
#         model_name = "Focal-T"
#     else:
#         pretrained_pth = "seem_focall_v0.pt"
#         model_name = "Focal-L"

#     if not os.path.exists(pretrained_pth):
#         os.system(
#             f"wget https://huggingface.co/xdecoder/SEEM/resolve/main/{pretrained_pth}"
#         )

#     print(f"Using model: {model_name}")

#     # build model
#     model = BaseModel(opt, build_model(opt)) \
#         .from_pretrained(pretrained_pth) \
#         .eval() \
#         .cuda()

#     # pre-compute text embeddings
#     with torch.no_grad():
#         model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
#             COCO_PANOPTIC_CLASSES + ["background"],
#             is_eval=True
#         )

#     # whisper (必要なら)
#     audio_model = whisper.load_model("base")

#     # load image
#     # 入力がディレクトリかファイルか判定
#     if os.path.isdir(args.input):
#         image_paths = sorted(
#             glob.glob(os.path.join(args.input, "*.png")) +
#             glob.glob(os.path.join(args.input, "*.jpg")) +
#             glob.glob(os.path.join(args.input, "*.jpeg"))
#         )
#     else:
#         image_paths = [args.input]
        
#     for img_path in image_paths:
#         pil_image = Image.open(img_path).convert("RGB")
#         image = {"image": pil_image, "mask": None}

#         # inference
#         # start_time = time.time()
        
#         with torch.no_grad():
#             with torch.autocast(device_type="cuda", dtype=torch.float16):
#                 tasks = ["Text"] if args.text else []

#                 result = interactive_infer_image(
#                     model,
#                     audio_model,
#                     image,
#                     tasks,
#                     None,          # refimg
#                     args.text,     # reftxt
#                     None,          # audio_pth
#                     None           # video_pth
#                 )

#         # print(f"Inference time: {time.time() - start_time:.4f} sec")
        
#         if isinstance(result, tuple):
#             result = result[0]

#         # 出力ファイル名を自動生成
#         base_name = os.path.splitext(os.path.basename(img_path))[0]
#         output_path = os.path.join(
#             os.path.dirname(args.output),
#             f"{base_name}_output.png"
#         )
#         result.save(output_path)
#         print(f"Saved to {output_path}")

#     # # result 保存
#     # if isinstance(result, tuple):
#     #     result = result[0]

#     # result.save(args.output)
#     # print(f"Saved to {args.output}")


# if __name__ == "__main__":
#     main()




import os
import torch
import argparse
from PIL import Image
import glob
import numpy as np

from modeling.BaseModel import BaseModel
from modeling import build_model
from utils.distributed import init_distributed
from utils.arguments import load_opt_from_config_files
from utils.constants import COCO_PANOPTIC_CLASSES
import time

# ----------------------------
# Argument
# ----------------------------
def parse_option():
    parser = argparse.ArgumentParser("SEEM Grounding Inference")
    parser.add_argument(
        "--conf_files",
        default="configs/seem/focall_unicl_lang_v1.yaml",
        help="path to config file",
    )
    parser.add_argument("--input", required=True, help="input image path or folder")
    parser.add_argument("--text", required=True, help="text prompt (e.g. road)")
    parser.add_argument("--output", default="output", help="output folder")
    return parser.parse_args()


# ----------------------------
# Main
# ----------------------------
def main():
    args = parse_option()

    os.makedirs(args.output, exist_ok=True)
    
    times = []

    # load config
    opt = load_opt_from_config_files([args.conf_files])
    opt = init_distributed(opt)

    # select pretrained model
    if "focalt" in args.conf_files:
        pretrained_pth = "seem_focalt_v1.pt"
        model_name = "Focal-T"
    else:
        pretrained_pth = "seem_focall_v1.pt"
        model_name = "Focal-L"

    if not os.path.exists(pretrained_pth):
        os.system(
            f"wget https://huggingface.co/xdecoder/SEEM/resolve/main/{pretrained_pth}"
        )

    print(f"Using model: {model_name}")

    # build model
    model = BaseModel(opt, build_model(opt)) \
        .from_pretrained(pretrained_pth) \
        .eval() \
        .cuda()

    # warmup text embeddings
    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            COCO_PANOPTIC_CLASSES + ["background"],
            is_eval=True
        )

    # image list
    if os.path.isdir(args.input):
        image_paths = sorted(
            glob.glob(os.path.join(args.input, "*.png")) +
            glob.glob(os.path.join(args.input, "*.jpg")) +
            glob.glob(os.path.join(args.input, "*.jpeg"))
        )
    else:
        image_paths = [args.input]

    for img_path in image_paths:

        pil_image = Image.open(img_path).convert("RGB")
        image_np = np.array(pil_image)

        # HWC → CHW tensor
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).cuda()

        batch_inputs = [{
            "image": image_tensor,
            "groundings": {
                "texts": [[args.text]]
            },
            "height": image_tensor.shape[1],
            "width": image_tensor.shape[2]
        }]

        # # inference
        # start = torch.cuda.Event(enable_timing=True)
        # end = torch.cuda.Event(enable_timing=True)
        
        # start.record()
        start_time = time.time()

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                results = model.model.evaluate_grounding(batch_inputs, mode=None)
        
        end_time = time.time() - start_time
        print(f"end_time: {end_time:.6f} sec")
        times.append(end_time)

        # end.record()
        # torch.cuda.synchronize()
        # elapsed_time = start.elapsed_time(end)

        # times.append(elapsed_time / 1000)
        # print(elapsed_time / 1000, 'sec.')

        mask = results[0]["grounding_mask"].cpu().numpy()
        
        # 余計なチャネル次元を削除
        if mask.ndim == 3:
            mask = mask[0]
        
        # mask = (mask > 0).astype(np.uint8) * 255

        # mask_pil = Image.fromarray(mask)

        # base_name = os.path.splitext(os.path.basename(img_path))[0]
        # output_path = os.path.join(args.output, f"{base_name}_grounding.png")
        # mask_pil.save(output_path)

        # print(f"Saved to {output_path}")
        
        mask = (mask > 0).astype(np.uint8)

        # ==============================
        # 元画像と合成
        # ==============================
        overlay = image_np.copy()

        # 赤色マスク
        overlay[mask == 1] = [128, 64, 128]

        alpha = 0.8  # 透明度（0〜1）
        blended = (image_np * (1 - alpha) + overlay * alpha).astype(np.uint8)

        result_pil = Image.fromarray(blended)

        # 保存
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(args.output, f"{base_name}_overlay.png")
        result_pil.save(output_path)

        print(f"Saved to {output_path}")
    if len(times) > 3:
        print(f"平均処理時間: {np.mean(times[5:]):.6f} sec.")
    else:
        print(f"平均処理時間: {np.mean(times):.6f} sec.")

if __name__ == "__main__":
    main()