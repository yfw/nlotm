import argparse
import contextlib
import math
import os
from datetime import datetime

from einops import rearrange
import torch
from torch.cuda.amp import GradScaler
import torchvision.utils as vutils
from args import add_common_args
from data import GlobDataset
from prior import get_prior_model
from nlotm import NlotmImageAutoEncoder
from torch.nn import DataParallel as DP
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import linear_warmup, seed

parser = argparse.ArgumentParser()
add_common_args(parser)

parser.add_argument("--model_batch_size", type=int, default=320)

# specify one of load_path or model_checkpoint_path
parser.add_argument("--load_path", default=None)
parser.add_argument("--model_checkpoint_path", default=None)
parser.add_argument('--data_path', default='datasets/clevr-easy/train/*.png')
parser.add_argument('--prior_data_dir', default='datasets/clevr-easy/')
parser.add_argument('--version', default='v1')
parser.add_argument(
    "--log_path",
    default="outputs/nlotm/logs",
)

parser.add_argument("--checkpoint_path", default="checkpoint.pt.tar")
parser.add_argument("--prior_load_path", default=None)

parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--lr_warmup_steps", type=int, default=30000)
parser.add_argument("--lr_half_life", type=int, default=250000)
parser.add_argument("--skip_lr_schedule", default=False, action="store_true")
parser.add_argument("--clip", type=float, default=0.05)
parser.add_argument("--clip_norm_type", type=float, default=None)
parser.add_argument("--epochs", type=int, default=500)
parser.add_argument("--use_dp", default=True, action="store_true")

parser.add_argument(
    "--prior_type",
    type=str,
    default="discrete_block_tf",
    choices=["discrete_block_tf", "dvae_tf"],
)
parser.add_argument("--prior_d_model", type=int, default=192)
parser.add_argument("--prior_num_heads", type=int, default=4)
parser.add_argument("--prior_num_decoder_layers", type=int, default=8)
parser.add_argument("--prior_dropout", type=float, default=0.1)
parser.add_argument("--prior_norm_first", default=False, action="store_true")

parser.add_argument("--generate_images", default=False, action="store_true")
parser.add_argument("--generate_image_dir", default="", type=str)
parser.add_argument("--generate_num_images", default=1000, type=int)
parser.add_argument("--val_only", default=False, action="store_true")
parser.add_argument("--downstream_data_type", type=str, default="") # "" or "dvae"
parser.add_argument("--checkpoint_freq", type=int, default=100000)

args = parser.parse_args()


seed(args.seed)

arg_str_list = ["{}={}".format(k, v) for k, v in vars(args).items()]
arg_str = "__".join(arg_str_list)
log_dir = os.path.join(args.log_path, datetime.today().isoformat())
writer = SummaryWriter(log_dir)
writer.add_text("hparams", arg_str)

train_dataset = GlobDataset(root=args.data_path, phase="train", img_size=args.image_size)
val_dataset = GlobDataset(root=args.data_path, phase="val", img_size=args.image_size)


train_sampler = None
val_sampler = None

loader_kwargs = {
    "batch_size": args.model_batch_size,
    "shuffle": False,
    "num_workers": args.num_workers,
    "pin_memory": True,
    "drop_last": False,
}

train_loader = DataLoader(train_dataset, sampler=train_sampler, **loader_kwargs)
val_loader = DataLoader(val_dataset, sampler=val_sampler, **loader_kwargs)

model = NlotmImageAutoEncoder(args)

if args.load_path is not None:
    checkpoint = torch.load(args.load_path, map_location="cpu")
    model.load_state_dict(checkpoint)
    print(f"loaded model from load_path:{args.load_path}")
elif args.model_checkpoint_path is not None:
    checkpoint = torch.load(args.model_checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    print(f"loaded model from model_checkpoint_path:{args.model_checkpoint_path}")
else:
    raise NotImplementedError()

model = model.cuda()
model.eval()

if not args.generate_images:
    def create_prior_dataset(data_loader, filename):
        if os.path.isfile(filename):
            print(f'Found prior data: {filename}')
            dataset = torch.load(filename, map_location="cpu")
            return dataset
        else:
            prior_data = []
            with torch.no_grad():
                model.eval()
                for x in tqdm(data_loader):
                    x = x.cuda()
                    # (B, D)
                    z = model.get_z(x)
                    prior_data.append(z.cpu())
            dataset = torch.cat(prior_data, dim=0)
            torch.save(dataset, filename)
            return dataset

    prior_train_filename = os.path.join(args.prior_data_dir, f"prior_train_{args.version}.pt")
    prior_train_data = create_prior_dataset(train_loader, prior_train_filename)
    prior_val_filename = os.path.join(args.prior_data_dir, f"prior_val_{args.version}.pt")
    prior_val_data = create_prior_dataset(val_loader, prior_val_filename)

    loader_kwargs["batch_size"] = args.batch_size
    loader_kwargs["drop_last"] = True
    loader_kwargs["shuffle"] = True
    prior_train_loader = DataLoader(prior_train_data, **loader_kwargs)

    loader_kwargs["shuffle"] = False
    prior_val_loader = DataLoader(prior_val_data, **loader_kwargs)

    train_epoch_size = len(prior_train_loader)
    val_epoch_size = len(prior_val_loader)

    log_interval = train_epoch_size // 5

################################
print("Done setting up data...")
################################

prior_model = get_prior_model(args)

if os.path.isfile(args.checkpoint_path):
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    start_epoch = checkpoint["epoch"]
    best_val_loss = checkpoint["best_val_loss"]
    best_epoch = checkpoint["best_epoch"]
    prior_model.load_state_dict(checkpoint["model"])
    print(f"loaded checkpoint:{args.checkpoint_path}")
else:
    checkpoint = None
    start_epoch = 0
    best_val_loss = math.inf
    best_epoch = 0

if args.prior_load_path is not None:
    prior_checkpoint = torch.load(args.prior_load_path, map_location="cpu")
    prior_model.load_state_dict(prior_checkpoint)
    print(f"loaded prior_load_path:{args.prior_load_path}")

prior_model = prior_model.cuda()
if args.use_dp:
    prior_model = DP(prior_model)

optimizer = Adam(
    params=prior_model.parameters(),
    lr=args.lr,
)

scaler = GradScaler()
if checkpoint is not None:
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])


def get_model(m):
    if args.use_dp:
        return m.module
    else:
        return m

if args.generate_images:
    assert args.generate_image_dir != "", "Need to specific --generate_image_dir"
    os.makedirs(args.generate_image_dir, exist_ok=True)
    with torch.no_grad():
        prior_model.eval()
        i = 0
        sample_size = 100
        # generate at least as many as generate_num_images
        num_batches = (args.generate_num_images + sample_size - 1) // sample_size
        for idx in range(num_batches):
            with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                z_gen = get_model(prior_model).sample(sample_size)
                recon = model.recon_z(z_gen)
            for image in recon:
                vutils.save_image(image, f'{args.generate_image_dir}/{i}.png')
                i += 1
            print(f'batch {idx}/{num_batches}')
        prior_model.train()

for epoch in range(start_epoch, args.epochs):
    prior_model.train()

    for batch, z in enumerate(prior_train_loader):
        if args.val_only:
            break

        global_step = epoch * train_epoch_size + batch

        if not args.skip_lr_schedule:
            lr_warmup_factor = linear_warmup(
                global_step, 0.0, 1.0, 0.0, args.lr_warmup_steps
            )
            lr_decay_factor = math.exp(global_step / args.lr_half_life * math.log(0.5))
            optimizer.param_groups[0]["lr"] = lr_decay_factor * lr_warmup_factor * args.lr

        z = z.to("cuda", non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
            loss = get_model(prior_model).loss(z)

            if args.use_dp:
                loss = loss.mean()

        if args.fp16:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
        else:
            loss.backward()
        clip_norm_type = (
            args.clip_norm_type if args.clip_norm_type is not None else "inf"
        )
        norm = clip_grad_norm_(prior_model.parameters(), args.clip, clip_norm_type)
        if args.fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        with torch.no_grad():
            if batch % log_interval == 0:
                print(
                    "Train Epoch: {:3} [{:5}/{:5}] \t Loss: {:F}".format(
                        epoch + 1, batch, train_epoch_size, loss.item()
                    )
                )

                prior_model.eval()
                with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                    # use val dataset here so we always get same set and we can match with underlying dataset
                    z = next(iter(prior_val_loader))
                    z = z.to("cuda", non_blocking=True)
                    z_pred = get_model(prior_model).get_z_for_recon(z)
                    recon = model.recon_z(z_pred)
                    grid = vutils.make_grid(recon, pad_value=0.5)
                    writer.add_image(
                        "TRAIN_recons/recons".format(epoch + 1), grid, global_step=epoch + 1
                    )

                    original = next(iter(DataLoader(Subset(val_loader.dataset, torch.arange(args.batch_size)), batch_size=args.batch_size)))
                    original_grid = vutils.make_grid(original, pad_value=0.5)
                    writer.add_image(
                        "TRAIN_recons/original".format(epoch + 1), original_grid, global_step=epoch + 1
                    )
                prior_model.train()

                writer.add_scalar("TRAIN/loss", loss.item(), global_step)
                writer.add_scalar(
                    "TRAIN/lr", optimizer.param_groups[0]["lr"], global_step
                )
                writer.add_scalar("TRAIN/norm", norm, global_step)

            if global_step > 0 and global_step % args.checkpoint_freq == 0:
                checkpoint = {
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "model": get_model(prior_model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict()
                }
                checkpoint_name = f"checkpoint_s{global_step}.pt.tar"
                torch.save(checkpoint, os.path.join(log_dir, checkpoint_name))
                print(f'saved intermediate checkpoint: {os.path.join(log_dir, checkpoint_name)}')

    with torch.no_grad():
        prior_model.eval()
        with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
            z_gen = get_model(prior_model).sample(64)
            recon = model.recon_z(z_gen)
        grid = vutils.make_grid(recon, pad_value=0.5)
        writer.add_image(
            "TRAIN_recons/samples".format(epoch + 1), grid, global_step=epoch + 1
        )
        prior_model.train()

    with torch.no_grad():
        prior_model.eval()

        total_val_loss = 0.0

        for batch, z in enumerate(prior_val_loader):
            z = z.to("cuda", non_blocking=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                val_loss = get_model(prior_model).loss(z)

            if args.use_dp:
                val_loss = val_loss.mean()

            total_val_loss += val_loss.item()

        total_val_loss /= val_epoch_size

        writer.add_scalar("VAL/loss", total_val_loss, epoch + 1)

        print("====> Epoch: {:3} \t Loss = {:F}".format(epoch + 1, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1

            torch.save(
                get_model(prior_model).state_dict(),
                os.path.join(log_dir, "best_model.pt"),
            )

            if 50 <= epoch:
                with torch.autocast(device_type='cuda', dtype=torch.float16) if args.fp16 else contextlib.nullcontext():
                    z_gen = get_model(prior_model).sample(64)
                    recon = model.recon_z(z_gen)
                grid = vutils.make_grid(recon, pad_value=0.5)
                writer.add_image(
                    "VAL_recons/samples".format(epoch + 1), grid, global_step=epoch + 1
                )

        writer.add_scalar("VAL/best_loss", best_val_loss, epoch + 1)

        checkpoint = {
            "epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "model": prior_model.module.state_dict()
            if args.use_dp
            else prior_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict()
        }

        torch.save(checkpoint, os.path.join(log_dir, "checkpoint.pt.tar"))
        print(f'saved checkpoint: {os.path.join(log_dir, "checkpoint.pt.tar")}')

        print("====> Best Loss = {:F} @ Epoch {}".format(best_val_loss, best_epoch))

writer.close()