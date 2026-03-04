
import argparse
from inference.Roaya_wrapper import RoayaVLWrapper

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="e.g. ./checkpoints/Roaya-VL-3B")
    parser.add_argument("--image", required=True, help="e.g. examples/Train_Pipeline.png")
    parser.add_argument("--prompt", required=True, help='e.g. "Describe the training pipeline of RoayaVL."')
    args = parser.parse_args()

    model = RoayaVLWrapper(args.model_path, device="cuda", verbose=True)
    pred = model.generate(args.prompt, images=[args.image], max_new_tokens=256, temperature=0.1)

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(pred)
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++')


if __name__ == "__main__":
    main()