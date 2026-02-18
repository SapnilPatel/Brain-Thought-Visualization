"""
Author: Neel Shah, Sapnil Patel, Yagnik Poshiya, Zijiao, Jack Qing, Patrick Finley
GitHub:  @neeldevenshah, @SapnilPatel, @yagnikposhiya, @zjc062, @jqin4749, @patrick-finley
Team name: ThreeMinds
Charotar University of Science and Technology

@InProceedings{Chen_2023_CVPR,
    author    = {Chen, Zijiao and Qing, Jiaxin and Xiang, Tiange and Yue, Wan Lin and Zhou, Juan Helen},
    title     = {Seeing Beyond the Brain: Masked Modeling Conditioned Diffusion Model for Human Vision Decoding},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
"""

from setuptools import setup, find_packages

setup(
    name='brain-tv',
    version='0.0.1',
    description='',
    packages=find_packages(),
    install_requires=[
        'torch',
        'numpy',
        'tqdm',
        'timm'
    ],
)