import fire
import time
import warnings

from multiprocessing import Queue, get_context
from RealtimeSTT import AudioToTextRecorder

from story_time.utils.viewer import receive_images
from story_time.utils.wrapper import StreamDiffusionWrapper

warnings.simplefilter(action="ignore", category=FutureWarning)


def speech_transcription_process(prompt_queue: Queue) -> None:
    recorder = AudioToTextRecorder()

    def parse_text(text: str):
        prompt_queue.put(text, block=False)
        print(f"{text}")

    while True:
        recorder.text(parse_text)


def image_generation_process(
    queue: Queue,
    fps_queue: Queue,
    prompt_queue: Queue,
    model_id_or_path: str,
) -> None:
    """
    Process for generating images based on a prompt using a specified model.

    Parameters
    ----------
    queue : Queue
        The queue to put the generated images in.
    fps_queue : Queue
        The queue to put the calculated fps.
    prompt : Queue
        The prompts to generate images from.
    model_id_or_path : str
        The name of the model to use for image generation.
    """
    stream = StreamDiffusionWrapper(
        model_id_or_path=model_id_or_path,
        t_index_list=[0],
        frame_buffer_size=1,
        warmup=10,
        use_lcm_lora=False,
        mode="txt2img",
        cfg_type="none",
        use_denoising_batch=True,
    )

    stream.prepare(
        prompt="line art drawing. professional, sleek, modern, minimalist, graphic, line art, vector graphics",
        num_inference_steps=50,
    )

    while True:
        start_time = time.time()
        if not prompt_queue.empty():
            stream.stream.update_prompt(
                f"{prompt_queue.get(block=False)}, line art drawing. professional, sleek, modern, minimalist, graphic, line art, vector graphics"
            )

        x_outputs = stream.stream(queue.get(block=False) if not queue.empty() else None).cpu()
        queue.put(x_outputs, block=False)

        fps = 1 / (time.time() - start_time)
        fps_queue.put(fps)


def main(
    model_id_or_path: str = "stabilityai/sd-turbo",
) -> None:
    """
    Main function to start the image generation and viewer processes.
    """
    ctx = get_context("spawn")
    queue = ctx.Queue()
    fps_queue = ctx.Queue()
    prompt_queue = ctx.Queue()

    process1 = ctx.Process(
        target=image_generation_process,
        args=(queue, fps_queue, prompt_queue, model_id_or_path),
    )
    process1.start()

    process2 = ctx.Process(target=receive_images, args=(queue, fps_queue))
    process2.start()

    process3 = ctx.Process(target=speech_transcription_process, args=(prompt_queue,))
    process3.start()

    process1.join()
    process2.join()


def test_streamdiffusion():
    fire.Fire(main)


if __name__ == "__main__":
    test_streamdiffusion()
