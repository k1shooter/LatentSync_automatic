import os
import subprocess
import argparse

def run_latentsync(
    input_video, input_audio, output_video, 
    unet_config="configs/unet/stage2_512.yaml",
    unet_ckpt="checkpoints/latentsync_unet.pt",
    steps=50, guidance_scale=1.5, enable_deepcache=True):

    temp_out = output_video + ".noaudio.mp4"
    latentsync_cmd = [
        "python", "-m", "scripts.inference",
        "--unet_config_path", unet_config,
        "--inference_ckpt_path", unet_ckpt,
        "--inference_steps", str(steps),
        "--guidance_scale", str(guidance_scale),
        "--video_path", input_video,
        "--audio_path", input_audio,
        "--video_out_path", temp_out,
    ]
    if enable_deepcache:
        latentsync_cmd.append("--enable_deepcache")
    print("\n[INFO] Start LatentSync inference ...")
    subprocess.run(latentsync_cmd, check=True)
    return temp_out

def merge_audio(latentsync_video, audio_file, final_output):
    merge_cmd = [
        "ffmpeg", "-i", latentsync_video, "-i", audio_file,
        "-c", "copy", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", final_output
    ]
    print("\n[INFO] Merging audio with ffmpeg ...")
    subprocess.run(merge_cmd, check=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--audio", required=True, help="Input audio path")
    parser.add_argument("--out", required=True, help="Output video path")
    parser.add_argument("--unet_config", default="configs/unet/stage2_512.yaml")
    parser.add_argument("--unet_ckpt", default="checkpoints/latentsync_unet.pt")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=1.5)
    parser.add_argument("--no_deepcache", action="store_true")
    args = parser.parse_args()

    temp_video = run_latentsync(
        input_video=args.video,
        input_audio=args.audio,
        output_video=args.out,
        unet_config=args.unet_config,
        unet_ckpt=args.unet_ckpt,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        enable_deepcache=not args.no_deepcache)

    merge_audio(latentsync_video=temp_video, audio_file=args.audio, final_output=args.out)

    if os.path.exists(temp_video):
        os.remove(temp_video)
    print(f"\n🎬 변환 및 립싱크+음성 합성 완료! -> {args.out}")

if __name__ == "__main__":
    main()
