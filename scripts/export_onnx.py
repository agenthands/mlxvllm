import torch
import os
from gui_actor.modeling import Qwen2VLForConditionalGenerationWithPointer
from transformers import AutoProcessor

def export_gui_actor(model_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading model: {model_id}")
    model = Qwen2VLForConditionalGenerationWithPointer.from_pretrained(
        model_id, torch_dtype=torch.float32, device_map="cpu"
    ).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    # 1. Export Vision Tower
    print("Exporting Vision Tower...")
    dummy_pixel_values = torch.randn(1, 3, 224, 224)
    dummy_grid_thw = torch.tensor([[1, 14, 14]])
    torch.onnx.export(
        model.visual,
        (dummy_pixel_values, dummy_grid_thw),
        os.path.join(output_dir, "vision_tower.onnx"),
        input_names=["pixel_values", "grid_thw"],
        output_names=["image_embeds"],
        dynamic_axes={"pixel_values": {0: "batch", 2: "height", 3: "width"}, "image_embeds": {0: "num_patches"}},
        opset_version=17
    )

    # 2. Export Pointer Head (The Action Head)
    print("Exporting Pointer Head...")
    d_model = model.config.hidden_size
    dummy_visual_hs = torch.randn(1, 100, d_model)
    dummy_target_hs = torch.randn(1, 1, d_model)
    torch.onnx.export(
        model.multi_patch_pointer_head,
        (dummy_visual_hs, dummy_target_hs),
        os.path.join(output_dir, "pointer_head.onnx"),
        input_names=["visual_hidden_states", "target_hidden_states"],
        output_names=["attn_weights", "loss"],
        dynamic_axes={"visual_hidden_states": {1: "num_patches"}, "target_hidden_states": {1: "num_targets"}},
        opset_version=17
    )

    # Note: Exporting the full 7B LLM requires careful memory handling.
    # In a production environment, you would typically use 'optimum-cli' for the LLM part.
    print(f"Components exported to {output_dir}. Use optimum-cli to export the Qwen2 LLM backbone separately.")

if __name__ == "__main__":
    import sys
    model_path = sys.argv[1] if len(sys.argv) > 1 else "microsoft/GUI-Actor-7B-Qwen2-VL"
    export_gui_actor(model_path, "onnx_models")
