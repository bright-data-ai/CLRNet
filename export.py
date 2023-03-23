import argparse

import torch
import torch.nn.parallel

from clrnet.models.registry import build_net
from clrnet.utils.config import Config
from clrnet.utils.net_utils import load_network


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    cfg.gpus = 1
    cfg.load_from = args.load_from
    net = build_net(cfg)
    load_network(net, cfg.load_from)
    net.eval()
    dummy_input = torch.randn(1, 3, cfg.img_h, cfg.img_w, requires_grad=True)
    traced_model = torch.jit.trace(net, dummy_input)
    torch.jit.save(traced_model, args.outfile)
    print(traced_model)
    print(f"model saved to {args.outfile}")


def parse_args():
    parser = argparse.ArgumentParser(description='Export checkpoint to onnx')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('outfile', help='onnx file path')
    parser.add_argument('--load_from',
                        default=None,
                        help='the checkpoint file to load from')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
