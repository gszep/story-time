from RealtimeSTT import AudioToTextRecorder
from diffusers import AutoPipelineForText2Image
import torch


def process_text(prompt):
    print(prompt)
    image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
    image.save("output.png")


pipe = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/sd-turbo", torch_dtype=torch.float16, variant="fp16"
)
pipe.to("cuda")


if __name__ == "__main__":
    print("Wait until it says 'speak now'")
    recorder = AudioToTextRecorder()

    while True:
        recorder.text(process_text)
