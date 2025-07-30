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

def merge_audio(latentsync_video, audio_file, merged_output_tmp):
    merge_cmd = [
        "ffmpeg", 
        "-i", latentsync_video, "-i", audio_file, "-c", "copy",
        "-map", "0:v:0", "-map", "1:a:0",
        "-shortest",
        merged_output_tmp
    ]
    print("\n[INFO] Merging audio with ffmpeg (aac re-encode)...")
    subprocess.run(merge_cmd, check=True)

def final_reencode_aac(input_file, output_file):
    reencode_cmd = [
        "ffmpeg", "-y",
        "-i", input_file,
        "-c:v", "copy",
        "-c:a", "aac",
        "-strict", "experimental",
        output_file
    ]
    print("\n[INFO] Final AAC audio re-encoding for compatibility ...")
    subprocess.run(reencode_cmd, check=True)

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
        enable_deepcache=not args.no_deepcache
    )

    # 1차 병합(오디오 aac로)
    merged_tmp = args.out + ".merged_tmp.mp4"
    merge_audio(latentsync_video=temp_video, audio_file=args.audio, merged_output_tmp=merged_tmp)

    # 2차: 혹시라도 PCM/비표준이 들어가면 AAC 재인코딩
    final_reencode_aac(merged_tmp, args.out)

    # 임시파일 정리
    for f in [temp_video, merged_tmp]:
        if os.path.exists(f):
            os.remove(f)
    print(f"\n🎬 변환 및 립싱크+음성 합성 완료! (aac 오디오 적용) -> {args.out}")

if __name__ == "__main__":
    main()
