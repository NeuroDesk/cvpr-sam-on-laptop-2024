import argparse
import warnings
import subprocess

import torch
from src.litemedsam.build_sam import sam_model_registry
from src.litemedsam.utils.sam_onnx import EncoderModel
from src.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
import src.efficient_sam.efficientsam_onnx as efficientsam_onnx


parser = argparse.ArgumentParser(
    description="Export the sam encoder to an onnx model."
)
parser.add_argument(
    "--checkpoint", type=str, required=True,
    help="The path to the sam model checkpoint.",
)
parser.add_argument(
    "--output", type=str, required=True,
    help="The filename to save the onnx model to.",
)
parser.add_argument(
    "--model-type", type=str, required=True,
    help="In ['default', 'vit_h', 'vit_l', 'vit_b', 'vit_t', 'vitt', 'vits']. Which type of SAM model to export.",
)
parser.add_argument(
    "--opset", type=int, default=17,
    help="The ONNX opset version to use. Must be >=11",
)
parser.add_argument(
    "--use-preprocess", action="store_true",
    help=("Embed pre-processing into the model",),
)
parser.add_argument(
    "--quantize-out", type=str, default=None,
    help=(
        "If set, will quantize the model and save it with this name. "
        "Quantization is performed with quantize_dynamic from "
        "onnxruntime.quantization.quantize."
    ),
)
parser.add_argument(
    "--gelu-approximate",
    action="store_true",
    help=(
        "Replace GELU operations with approximations using tanh. Useful "
        "for some runtimes that have slow or unimplemented erf ops, used in GELU."
    ),
)



def optimize(onnx_model_path, optimized_model_path):
    command = "python -m onnxoptimizer " + onnx_model_path + " " + optimized_model_path
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    process.wait()

def run_export(
    model_type: str,
    checkpoint: str,
    output: str,
    use_preprocess: bool,
    opset: int,
    gelu_approximate: bool = False,
) -> None:
    print("Loading model...")
    # build model
    if model_type == 'vits':
        sam = build_efficient_sam_vits(checkpoint)
        onnx_model = efficientsam_onnx.OnnxEfficientSamEncoder(model=sam)
    if model_type == 'vitt':
        sam = build_efficient_sam_vitt(checkpoint)
        onnx_model = efficientsam_onnx.OnnxEfficientSamEncoder(model=sam)
    else:
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        onnx_model = EncoderModel(
            model=sam,
            use_preprocess=use_preprocess,
        )

    if gelu_approximate:
        for _, m in onnx_model.named_modules():
            if isinstance(m, torch.nn.GELU):
                m.approximate = "tanh"

    image_size = [256, 256]
    if use_preprocess:
        dummy_input = {
            "input_image": torch.randint(
                0, 255, (image_size[0], image_size[1], 3), dtype=torch.uint8
            )
        }
        dynamic_axes = {
            "input_image": {0: "image_height", 1: "image_width"},
        }
    else:
        dummy_input = {
            "input_image": torch.randn(
                (1, 3, image_size[0], image_size[1]), dtype=torch.float
            )
        }
        dynamic_axes = None
    _ = onnx_model(**dummy_input)

    output_names = ["image_embeddings"]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        print(f"Exporting onnx model to {output}...")
        with open(output, "wb") as f:
            torch.onnx.export(
                onnx_model,
                tuple(dummy_input.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=opset,
                do_constant_folding=True,
                input_names=list(dummy_input.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    run_export(
        model_type=args.model_type,
        checkpoint=args.checkpoint,
        output=args.output,
        use_preprocess=args.use_preprocess,
        opset=args.opset,
        gelu_approximate=args.gelu_approximate,
    )

    optimize(args.output, args.output.replace(".onnx", ".opt.onnx"))

    if args.quantize_out is not None:
        from onnxruntime.quantization import QuantType  # type: ignore
        from onnxruntime.quantization.quantize import quantize_dynamic  # type: ignore

        print(f"Quantizing model and writing to {args.quantize_out}...")
        quantize_dynamic(
            model_input=args.output.replace(".onnx", ".opt.onnx"),
            model_output=args.quantize_out,
            # optimize_model=True,
            per_channel=True,
            reduce_range=True,
            weight_type=QuantType.QUInt8,
            nodes_to_quantize=["MatMul", "Add"],
            # nodes_to_exclude=["Slice"]
        )
        print("Done!")