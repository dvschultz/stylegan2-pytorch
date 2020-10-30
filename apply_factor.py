import argparse
import os
import subprocess

import torch
from torchvision import utils

from model import Generator


if __name__ == "__main__":
    torch.set_grad_enabled(False)

    parser = argparse.ArgumentParser(description="Apply closed form factorization")

    parser.add_argument(
        "-i", "--index", type=int, default=0, help="index of eigenvector"
    )
    parser.add_argument(
        "-d",
        "--degree",
        type=float,
        default=5,
        help="scalar factors for moving latent vectors along eigenvector",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help='channel multiplier factor. config-f = 2, else = 1',
    )
    parser.add_argument("--ckpt", type=str, required=True, help="stylegan2 checkpoints")
    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "-n", "--n_sample", type=int, default=7, help="number of samples created"
    )
    parser.add_argument(
        "--truncation", type=float, default=0.7, help="truncation factor"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to run the model"
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="factor",
        help="filename prefix to result samples",
    )
    parser.add_argument(
        "factor",
        type=str,
        help="name of the closed form factorization result factor file",
    )
    parser.add_argument(
        "--vid_increment", type=float, default=0.1, help="increment degree for interpolation video"
    )
    vid_parser = parser.add_mutually_exclusive_group(required=False)
    vid_parser.add_argument('--video', dest='vid', action='store_true')
    vid_parser.add_argument('--no-video', dest='vid', action='store_false')
    vid_parser.set_defaults(vid=False)

    args = parser.parse_args()

    eigvec = torch.load(args.factor)["eigvec"].to(args.device)
    ckpt = torch.load(args.ckpt)
    g = Generator(args.size, 512, 8, channel_multiplier=args.channel_multiplier).to(args.device)
    g.load_state_dict(ckpt["g_ema"], strict=False)

    trunc = g.mean_latent(4096)

    latent = torch.randn(args.n_sample, 512, device=args.device)
    latent = g.get_latent(latent)

    direction = args.degree * eigvec[:, args.index].unsqueeze(0)

    img, _ = g(
        [latent],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img1, _ = g(
        [latent + direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )
    img2, _ = g(
        [latent - direction],
        truncation=args.truncation,
        truncation_latent=trunc,
        input_is_latent=True,
    )

    grid = utils.save_image(
        torch.cat([img1, img, img2], 0),
        f"{args.out_prefix}_index-{args.index}_degree-{args.degree}.png",
        normalize=True,
        range=(-1, 1),
        nrow=args.n_sample,
    )

    if(args.vid):
        count = 0
        for l in latent:
            fname = f"{args.out_prefix}_index-{args.index}_degree-{args.degree}_index-{count}"
            if not os.path.exists(fname):
                os.makedirs(fname)
            d = -1*args.degree

            fcount = 0
            while d < args.degree:
                # generate latent
                dir = d * eigvec[:, args.index].unsqueeze(0)
                img, _ = g(
                    [l + dir],
                    truncation=args.truncation,
                    truncation_latent=trunc,
                    input_is_latent=True,
                )

                # generate latent
                grid = utils.save_image(
                    img,
                    f"{fname}/{fname}_{fcount:04}.png",
                    normalize=True,
                    range=(-1, 1),
                    nrow=1,
                )

                d+=args.vid_increment
                fcount+=1


            cmd=f"ffmpeg -y -r 24 -i {fname}/{fname}_%04d.png -vcodec libx264 -pix_fmt yuv420p {fname}/{fname}.mp4"
            subprocess.call(cmd, shell=True)

            count+=1





            
