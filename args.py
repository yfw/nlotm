def add_common_args(parser):
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--image_channels", type=int, default=3)

    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--num_slots", type=int, default=4)
    parser.add_argument("--num_blocks", type=int, default=16)
    parser.add_argument("--cnn_hidden_size", type=int, default=512)
    parser.add_argument("--slot_size", type=int, default=2048)
    parser.add_argument("--mlp_hidden_size", type=int, default=192)
    parser.add_argument("--num_prototypes", type=int, default=64)

    parser.add_argument("--vocab_size", type=int, default=4096)
    parser.add_argument("--num_decoder_layers", type=int, default=8)
    parser.add_argument("--num_decoder_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=192)
    parser.add_argument("--dropout", type=int, default=0.1)

    parser.add_argument("--fp16", default=False, action="store_true")

    parser.add_argument(
        "--vq_type",
        type=str,
        default="vq_ema_dcr",
        choices=[
            "vq_ema_dcr",
            "none",
        ],
    )
    parser.add_argument("--vq_beta", type=float, default=1.0)
    parser.add_argument("--commitment_beta", type=float, default=50.0)

    parser.add_argument(
        "--slot_init_type", type=str, default="random", choices=["random", "learned"]
    )
