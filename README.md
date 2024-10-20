Image upscaler using SDGAN 
- https://arxiv.org/pdf/1609.04802

TODO:
- Build out SRGAN (wip)
- Light training, will do full training during super-offpeak hours.
- Build out backend in FastAPI
- Containerize
- Deploy on cluster (?) 
    - ^ test performance on CPU first, might be better to localhost w/ my GPU.

Future Considerations:
- Super-Resolving Images : https://arxiv.org/abs/2410.12961
- Hybrid Attention Separable Network for Super Resolution: https://arxiv.org/abs/2410.09844
