import subprocess
from argparse import ArgumentParser
from os.path import abspath
from shutil import rmtree
from sys import stderr
from typing import Literal, Optional
from uuid import uuid1

from RkllmToolkitCli.api import RKLLMBase


class HuggingfaceConvertor:
    def __init__(self, repository: str):
        self._directory = f'/tmp/{uuid1()}'
        self._repository_url = f'https://huggingface.co/{repository}'
        self._builder = RKLLMBase()

    def download_model(self) -> bool:
        return subprocess.run(['git', 'clone', self._repository_url, self._directory]).returncode == 0

    def load_model(self, device: Literal['cpu', 'cuda'] = 'cpu') -> bool:
        return self._builder.load_huggingface(self._directory, device=device) == 0

    def build_model(self, disable_quantization: bool = False, disable_optimization: bool = False,
                    quantized_dtype: str = 'w8a8', target_platform: Literal['rk3576', 'rk3588'] = 'rk3588') -> bool:
        return self._builder.build(do_quantization=not disable_quantization,
                                   optimization_level=not disable_optimization, quantized_dtype=quantized_dtype,
                                   target_platform=target_platform) == 0

    def export_model(self, output: Optional[str] = None) -> bool:
        if output is None:
            output = f'{self._repository_url.split("/")[-1]}.rkllm'

        return self._builder.export_rkllm(abspath(output)) == 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        rmtree(self._directory)


def main():
    parser = ArgumentParser(
        description='''
        Convert Huggingface model to RKLLM model.
        
        Example: toolkit microsoft/Phi-3-mini-4k-instruct
        ''',
        epilog='''
        This program under the AGPL-3.0 license.
        Copyright (C) 2024 docjyJ
        '''
    )
    parser.add_argument('-o', '--output', help='output model file', metavar='MODEL_FILE', type=str)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='device type', metavar='DEVICE',
                        type=str)
    parser.add_argument('--no-quantization', action='store_true', help='disable quantization')
    parser.add_argument('--no-optimization', action='store_true', help='disable optimization')
    parser.add_argument('--quantized-type', default='w8a8', help='quantized data type', metavar='DTYPE', type=str)
    parser.add_argument('--target', choices=['rk3576', 'rk3588'], default='rk3588', help='target platform',
                        metavar='PLATFORM', type=str)
    parser.add_argument('repository', help='model huggingface repository', metavar='REPOSITORY', type=str)
    args = parser.parse_args()

    with HuggingfaceConvertor(args.repository) as convertor:
        if not convertor.download_model():
            print('ERROR - Download model failed', file=stderr)
            exit(-1)

        if not convertor.load_model(device=args.device):
            print('ERROR - Load model failed', file=stderr)
            exit(-1)

        if not convertor.build_model(disable_quantization=args.no_quantization,
                                     disable_optimization=args.no_optimization, quantized_dtype=args.quantized_type,
                                     target_platform=args.target):
            print('ERROR - Build model failed', file=stderr)
            exit(-1)

        if not convertor.export_model(args.output):
            print('ERROR - Export model failed', file=stderr)
            exit(-1)


if __name__ == '__main__':
    main()
