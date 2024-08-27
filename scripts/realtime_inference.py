import argparse
from omegaconf import OmegaConf
from .avatar import Avatar


if __name__ == "__main__":
    '''
    This script is used to simulate online chatting and applies necessary pre-processing such as face detection and face parsing in advance. During online chatting, only UNet and the VAE decoder are involved, which makes MuseTalk real-time.
    '''
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", 
                        type=str, 
                        default="configs/inference/realtime.yaml",
    )
    parser.add_argument("--fps", 
                        type=int, 
                        default=25,
    )
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=4,
    )
    parser.add_argument("--skip_save_images",
                        action="store_true",
                        help="Whether skip saving images for better generation speed calculation",
    )

    args = parser.parse_args()
    
    inference_config = OmegaConf.load(args.inference_config)
    print(inference_config)
    
    
    for avatar_id in inference_config:
        data_preparation = inference_config[avatar_id]["preparation"]
        video_path = inference_config[avatar_id]["video_path"]
        bbox_shift = inference_config[avatar_id]["bbox_shift"]
        avatar = Avatar(
            avatar_id = avatar_id, 
            video_path = video_path, 
            bbox_shift = bbox_shift, 
            batch_size = args.batch_size,
            preparation= data_preparation)
        
        audio_clips = inference_config[avatar_id]["audio_clips"]
        for audio_num, audio_path in audio_clips.items():
            print("Inferring using:",audio_path)
            avatar.inference(audio_path, 
                             audio_num, 
                             args.fps,
                             args.skip_save_images)
